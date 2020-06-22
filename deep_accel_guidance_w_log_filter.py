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

# Signal Library
from scipy import interpolate
from scipy import zeros, signal, random
from scipy import linalg as la

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")
    parser.add_argument("-fi", "--follower_id", dest='follower_id', default=0, type=int, help="Follower aircraft ID")
    parser.add_argument("-f", "--filename", dest='log_filename', default='log_000', type=str, help="Log file name")
    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = args.follower_id
    log_filename = args.log_filename
    max_duration = 100000
    log_placeholder = np.zeros((max_duration, 30))
    i=0 # for log increment

    ## Prepare Filter Specs : ##
    fs = 5        # Sampling frequency
    order = 2
    cutoff = 2.0
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff)
    dgn = dge = dgd = signal.lfilter_zi(b, a)
    


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
            last_target_yaw = 0.0
            total_time = 0.0
            
            if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:                    
                # Create state-augmentation queue (holds previous actions)
                past_actions = queue.Queue(maxsize = Settings.AUGMENT_STATE_WITH_ACTION_LENGTH)
        
                # Fill it with zeros to start
                for i in range(Settings.AUGMENT_STATE_WITH_ACTION_LENGTH):
                    past_actions.put(np.zeros(Settings.ACTION_SIZE), False)
            
            if Settings.AUGMENT_STATE_WITH_STATE_LENGTH > 0: 
                # Create state-augmentation queue (holds previous raw total states)
                past_states = queue.Queue(maxsize = Settings.AUGMENT_STATE_WITH_STATE_LENGTH)
                
                # Fill it with zeros to start
                for i in range(Settings.AUGMENT_STATE_WITH_STATE_LENGTH):
                    past_states.put(np.zeros(Settings.TOTAL_STATE_SIZE), False)
                
            while True:
                # TODO: make better frequency managing
                sleep(g.step)
                total_time = total_time + g.step
                # print('G IDS : ',g.ids) # debug....
                policy_input = np.zeros(Settings.TOTAL_STATE_SIZE) # initializing policy input
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + g.step
                    
                    
                    """ policy_input is: [chaser_x, chaser_y, chaser_z, chaser_theta, target_x, target_y, target_z, target_theta, 
                                          chaser_x_dot, chaser_y_dot, chaser_z_dot, chaser_theta_dot + (optional past action data)] 
                    
                    Note: chaser_theta_dot can be returned as a zero (it is discarded before being run through the network)
                    """
                    
                                    
                    #print('rc.W',rc.W)  # example to see the positions, or you can get the velocities as well...
                    if rc.id == target_id: # we've found the target
                        policy_input[3] =  rc.X[0] # target X [north] =   North
                        policy_input[4] = -rc.X[1] # targey Y [west]  = - East
                        policy_input[5] =  rc.X[2] # target Z [up]    =   Up
                        policy_input[6] =  np.unwrap([last_target_yaw, -rc.W[2]])[1] # target yaw  [counter-clockwise] = -yaw [clockwise]
                        last_target_yaw = policy_input[6]
                        #print("Target position: X: %.2f; Y: %.2f; Z: %.2f; Att %.2f" %(rc.X[0], -rc.X[1], rc.X[2], -rc.W[2]))
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                    if rc.id == follower_id: # we've found the chaser (follower)
                        policy_input[0] =  rc.X[0] # chaser X [north] =   North
                        policy_input[1] = -rc.X[1] # chaser Y [west]  = - East
                        policy_input[2] =  rc.X[2] # chaser Z [up]    =   Up                        
                        
                        policy_input[7] =  rc.V[0] # chaser V_x [north] =   North
                        policy_input[8] = -rc.V[1] # chaser V_y [west]  = - East
                        policy_input[9] =  rc.V[2] # chaser V_z [up]    =   Up
                        
                        #print("Time: %.2f; Chaser position: X: %.2f; Y: %.2f; Z: %.2f; Att %.2f; Vx: %.2f; Vy: %.2f; Vz: %.2f" %(rc.timeout, rc.X[0], -rc.X[1], rc.X[2], -rc.W[2], rc.V[0], -rc.V[1], rc.V[2]))
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                        
                # Save raw policy input incase we want to augment state with it
                raw_policy_input = policy_input
                    
                # Augment state with past action data if applicable
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:                        
                    past_action_data = np.asarray(past_actions.queue).reshape([-1]) # past actions reshaped into a column
                    
                    # Remove the oldest entry from the action log queue
                    past_actions.get(False)
                    
                    # Concatenate past actions to the policy input
                    policy_input = np.concatenate([policy_input, past_action_data])
                    
                if Settings.AUGMENT_STATE_WITH_STATE_LENGTH > 0:
                    past_state_data = np.asarray(past_states.queue).reshape([-1]) # past actions reshaped into a column
                    
                    # Remove the oldest entry from the state log queue
                    past_states.get(False)
                    
                    # Add current policy input to past_states so they'll be included in the augmented state next timestep
                    past_states.put(raw_policy_input, False)
                    
                    # Concatenate past states to the policy input
                    policy_input = np.concatenate([policy_input, past_state_data])
                    
                    
                    
                ############################################################
                ##### Received data! Process it and return the result! #####
                ############################################################
        	    # Calculating the proper policy input (deleting irrelevant states and normalizing input)
                # Normalizing
                if Settings.NORMALIZE_STATE:
                    normalized_policy_input = (policy_input - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE
                else:
                    normalized_policy_input = policy_input
        
                # Discarding irrelevant states
                normalized_policy_input = np.delete(normalized_policy_input, Settings.IRRELEVANT_STATES)
        
                # Reshaping the input
                normalized_policy_input = normalized_policy_input.reshape([-1, Settings.OBSERVATION_SIZE])
        
                # Run processed state through the policy
                deep_guidance = sess.run(actor.action_scaled, feed_dict={state_placeholder:normalized_policy_input})[0]
                # deep guidance = [ chaser_angular_velocity [counter-clockwise looking down from above], chaser_x_acceleration [north], chaser_y_acceleration [west], chaser_z_acceleration [up] ]
                
                # Adding the action taken to the past_action log
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:
                    past_actions.put(deep_guidance)
                    
                # Limit guidance commands if velocity is too high!
                # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
                current_velocity = policy_input[7:]
                deep_guidance[np.concatenate((np.array([False]), (np.abs(current_velocity) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance[1:]) == np.sign(current_velocity))))] = 0 
                
                deep_guidance[0], dgn = signal.lfilter(b, a, [deep_guidance[0]] , zi=dgn)
                deep_guidance[1], dge = signal.lfilter(b, a, [deep_guidance[1]] , zi=dge)
                deep_guidance[2], dgd = signal.lfilter(b, a, [deep_guidance[2]] , zi=dgd)
                
                # deep_guidance[2], deep_guidance_prev_filt[2] = signal.lfilter(b, a, deep_guidance[2], zi=deep_guidance_prev_filt[2])
                # deep_guidance[3], deep_guidance_prev_filt[3] = signal.lfilter(b, a, deep_guidance[3], zi=deep_guidance_prev_filt[3])
                # Send velocity/acceleration command to aircraft!
                #g.move_at_ned_vel( yaw=-deep_guidance[0])
                g.accelerate(north = deep_guidance[0], east = -deep_guidance[1], down = -deep_guidance[2])
                #print("Deep guidance command: a_x: %.2f; a_y: %.2f; a_z: %.2f" %( deep_guidance[1], deep_guidance[2], deep_guidance[3]))
                print("Time: %.2f; X: %.2f; Vx: %.2f; Ax: %.2f" %(total_time, policy_input[0], policy_input[7], deep_guidance[0]))
                print('Deep Guidance :',deep_guidance)

                # print('Policy input shape :',normalized_policy_input.shape)

                # Log all niput and outputs:
                t = time.time()-start_time
                log_placeholder[i,0] = t
                log_placeholder[i,1:4] = deep_guidance
                # log_placeholder[i,5:8] = deep_guidance_xf, deep_guidance_yf, deep_guidance_zf
                log_placeholder[i,4:4+len(normalized_policy_input[0])] = normalized_policy_input
                i += 1
    


        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            g.set_nav_mode()
            g.shutdown()
            sleep(0.2)
            with open(log_filename+".txt", 'wb') as f:
                np.save(f, log_placeholder[:i])
            exit()

if __name__ == '__main__':
    main()

#EOF