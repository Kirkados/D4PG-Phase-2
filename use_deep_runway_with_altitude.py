#!/usr/bin/env python

from __future__ import print_function

import sys
from os import path, getenv

from math import radians
from time import sleep
import numpy as np
import queue

# Deep guidance stuff
import tensorflow as tf
import time

from settings import Settings
from build_neural_networks import BuildActorNetwork

# Paparazzi guidance api
from guidance_common import Rotorcraft , Guidance



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")
    parser.add_argument("-fi", "--follower_id", dest='follower_id', default=0, type=int, help="Follower aircraft ID")
    parser.add_argument("-f", "--filename", dest='log_filename', default='log_runway_000', type=str, help="Log file name")
    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = args.follower_id
    log_filename = args.log_filename
    max_duration = 100000
    log_placeholder = np.zeros((max_duration, 100))
    i = 0 # for log increment
    
    ### Deep guidance initialization stuff
    tf.reset_default_graph()

    # Initialize Tensorflow, and load in policy
    with tf.Session() as sess:
        # Building the policy network
        state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.OBSERVATION_SIZE], name = "state_placeholder")
        actor = BuildActorNetwork(state_placeholder, scope='learner_actor_main')
    
        # Loading in trained network weights
        print("Attempting to load in previously-trained model\n")
        saver = tf.train.Saver() # initialize the tensorflow Saver()
    
        # Try to load in policy network parameters
        try:
            ckpt = tf.train.get_checkpoint_state('../')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\nModel successfully loaded!\n")
    
        except (ValueError, AttributeError):
            print("No model found... quitting :(")
            raise SystemExit
    
        #######################################################################
        ### Guidance model is loaded, now get data and run it through model ###
        #######################################################################


        try:
            start_time = time.time()
            g = Guidance(interface=interface, target_id=target_id, follower_id=follower_id)
            sleep(0.1)
            # g.set_guided_mode()
            sleep(0.2)

            total_time = 0.0
            
            last_deep_guidance = np.zeros(Settings.ACTION_SIZE)
            
            if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:                    
                # Create state-augmentation queue (holds previous actions)
                past_actions = queue.Queue(maxsize = Settings.AUGMENT_STATE_WITH_ACTION_LENGTH)
        
                # Fill it with zeros to start
                for j in range(Settings.AUGMENT_STATE_WITH_ACTION_LENGTH):
                    past_actions.put(np.zeros([Settings.NUMBER_OF_QUADS, Settings.ACTION_SIZE]), False)
            
            runway_state = np.zeros([Settings.RUNWAY_LENGTH_ELEMENTS, Settings.RUNWAY_WIDTH_ELEMENTS])
            last_runway_state = np.zeros([Settings.RUNWAY_LENGTH_ELEMENTS, Settings.RUNWAY_WIDTH_ELEMENTS])
                
            while True:
                # TODO: make better frequency managing
                sleep(g.step)
                total_time = total_time + g.step
                
                # Initializing quadrotor positions and velocities
                quad_positions = np.zeros([Settings.NUMBER_OF_QUADS, 3]) 
                quad_velocities = np.zeros([Settings.NUMBER_OF_QUADS, 3])
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + g.step
                    
                    
                    """ policy_input is: [chaser_x, chaser_y, chaser_z, target_x, target_y, target_z, target_theta, 
                                          chaser_x_dot, chaser_y_dot, chaser_z_dot, (optional past action data)] 
                    """

                    quad_number = rc.id - 1
                    try:
                        # Extracting position
                        quad_positions[ quad_number, 0] =  rc.X[0]
                        quad_positions[ quad_number, 1] = -rc.X[1]
                        quad_positions[ quad_number, 2] =  rc.X[2]
                        
                        # Extracting velocity
                        quad_velocities[quad_number, 0] =  rc.V[0]
                        quad_velocities[quad_number, 1] = -rc.V[1]
                        quad_velocities[quad_number, 2] =  rc.V[2]
                    except:
                        print("The quad IDs must start at 1 and increase from there!")
                        raise SystemExit
                
                # Check runway state
                # The size of each runway grid element
                each_runway_length_element = Settings.RUNWAY_LENGTH/Settings.RUNWAY_LENGTH_ELEMENTS
                each_runway_width_element  = Settings.RUNWAY_WIDTH/Settings.RUNWAY_WIDTH_ELEMENTS
                
                # Which zones is each quad in?
                rows = np.floor(quad_positions[:,0]/each_runway_length_element).astype(int)
                columns = np.floor(quad_positions[:,1]/each_runway_width_element).astype(int)
        
                # Which zones are actually over the runway?
                elements_to_keep = np.array((rows >= 0) & (rows < Settings.RUNWAY_LENGTH_ELEMENTS) & (columns >= 0) & (columns < Settings.RUNWAY_WIDTH_ELEMENTS) & (quad_positions[:,2] >= Settings.MINIMUM_CAMERA_ALTITUDE) & (quad_positions[:,2] <= Settings.MAXIMUM_CAMERA_ALTITUDE))
                
                # Removing runway elements that are not over the runway
                rows = rows[elements_to_keep]
                columns = columns[elements_to_keep]
        
                # Mark the visited tiles as explored
                runway_state[rows,columns] = 1
                #print(runway_state,last_runway_state)
                
                if np.any(last_runway_state != runway_state):
                    print("Runway elements discovered %i/%i" %(np.sum(runway_state), Settings.RUNWAY_LENGTH_ELEMENTS*Settings.RUNWAY_WIDTH_ELEMENTS))
                    
                    # Draw a new runway
                    print(np.flip(runway_state))
                
                if np.all(runway_state) == 1:
                    print("Explored the entire runway in %.2f seconds--Congratualtions! Quitting deep guidance" %(time.time()-start_time))
                    sys.exit()
                
                total_states = []
                # Building NUMBER_OF_QUADS states
                for j in range(Settings.NUMBER_OF_QUADS):
                    # Start state with your own 
                    this_quads_state = np.concatenate([quad_positions[j,:], quad_velocities[j,:]])               
                    # Add in the others' states, starting with the next quad and finishing with the previous quad
                    for k in range(j + 1, Settings.NUMBER_OF_QUADS + j):
                        this_quads_state = np.concatenate([this_quads_state, quad_positions[k % Settings.NUMBER_OF_QUADS,:], quad_velocities[k % Settings.NUMBER_OF_QUADS,:]])
                    
                    # All quad data is included, now append the runway state and save it to the total_state
                    total_states.append(this_quads_state)
                    
                    
                # Augment total_state with past actions, if appropriate
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:
                    # total_states = [Settings.NUMBER_OF_QUADS, Settings.TOTAL_STATE_SIZE]
                    # Just received a total_state from the environment, need to augment 
                    # it with the past action data and return it
                    # The past_action_data is of shape [Settings.AUGMENT_STATE_WITH_ACTION_LENGTH, Settings.NUMBER_OF_QUADS, Settings.TOTAL_STATE_SIZE]
                    # I swap the first and second axes so that I can reshape it properly
            
                    past_action_data = np.swapaxes(np.asarray(past_actions.queue),0,1).reshape([Settings.NUMBER_OF_QUADS, -1]) # past actions reshaped into rows for each quad     
                    total_states = np.concatenate([np.asarray(total_states), past_action_data], axis = 1)
            
                    # Remove the oldest entry from the action log queue
                    past_actions.get(False)
                
                # Concatenating the runway to the augmented state
                total_states = np.concatenate([total_states, np.tile(runway_state.reshape(-1),(Settings.NUMBER_OF_QUADS,1))], axis = 1)

                # Normalize the state
                if Settings.NORMALIZE_STATE:
                    total_states = (total_states - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE

                # Discarding irrelevant states
                observations = np.delete(total_states, Settings.IRRELEVANT_STATES, axis = 1)
                
                # Run processed state through the policy
                deep_guidance = sess.run(actor.action_scaled, feed_dict={state_placeholder:observations}) # deep guidance = [ chaser_x_acceleration [north], chaser_y_acceleration [west], chaser_z_acceleration [up] ]
                
                # Adding the action taken to the past_action log
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:
                    past_actions.put(deep_guidance)

                # Limit guidance commands if velocity is too high!
                # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
                for j in range(Settings.NUMBER_OF_QUADS):              
                    deep_guidance[j,(np.abs(quad_velocities[j,:]) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance[j,:]) == np.sign(quad_velocities[j,:]))] = 0 
        
                average_deep_guidance = (last_deep_guidance + deep_guidance)/2.0
                last_deep_guidance = deep_guidance
                last_runway_state = np.copy(runway_state)
                #last_runway_state = runway_state
                
                # Send velocity/acceleration command to aircraft!
                #g.accelerate(north = deep_guidance[0], east = -deep_guidance[1], down = -deep_guidance[2])
                
                # Get each quad to accelerate appropriately
                for j in range(Settings.NUMBER_OF_QUADS):
                    #g.accelerate(north = average_deep_guidance[j,0], east = -average_deep_guidance[j,1], down = -average_deep_guidance[j,2], quad_id = j + 1) # Averaged
                    g.accelerate(north = deep_guidance[j,0], east = -deep_guidance[j,1], down = -deep_guidance[j,2], quad_id = j + 1) # Raw
                    #g.accelerate(north = deep_guidance[j,0], east = -deep_guidance[j,1], down = 0, quad_id = j + 1) # Raw
                
                # Log all input and outputs:
                t = time.time()-start_time
                log_placeholder[i,0] = t
                log_placeholder[i,1:3*Settings.NUMBER_OF_QUADS + 1] = deep_guidance.reshape(-1)
                # log_placeholder[i,5:8] = deep_guidance_xf, deep_guidance_yf, deep_guidance_zf
                log_placeholder[i,3*Settings.NUMBER_OF_QUADS + 1:3*Settings.NUMBER_OF_QUADS + 1 + Settings.OBSERVATION_SIZE] = observations[0,:]
                i += 1
    


        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            g.set_nav_mode()
            g.shutdown()
            sleep(0.2)
            print("Saving file as %s.txt..." %(log_filename))
            with open(log_filename+".txt", 'wb') as f:
                np.save(f, log_placeholder[:i])
            print("Done!")
            exit()


if __name__ == '__main__':
    main()

#EOF