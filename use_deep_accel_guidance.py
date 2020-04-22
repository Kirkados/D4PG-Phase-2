#!/usr/bin/env python

from __future__ import print_function

import sys
from os import path, getenv

from math import radians
from time import sleep
import numpy as np

# Deep guidance stuff
import tensorflow as tf
import time

from settings import Settings
from build_neural_networks import BuildActorNetwork

# Paparazzi guidance api
from guidance_paparazzi.guidance_common import Rotorcraft , Guidance


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument("-ti", "--target_id", dest='target_id', default=0, type=int, help="Target aircraft ID")
    parser.add_argument("-fi", "--follower_id", dest='follower_id', default=0, type=int, help="Follower aircraft ID")
    args = parser.parse_args()

    interface = None
    target_id = args.target_id
    follower_id = args.follower_id
    
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
            g = Guidance(interface=interface, target_id=target_id, follower_id=follower_id)
            sleep(0.1)
            # g.set_guided_mode()
            sleep(0.2)
            last_target_yaw = 0.0
            last_chaser_yaw = 0.0    
            while True:
                # TODO: make better frequency managing
                sleep(g.step)
                # print('G IDS : ',g.ids) # debug....
                policy_input = np.zeros(12) # initializing policy input
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + g.step
                    
                    
                    """ policy_input is: [chaser_x, chaser_y, chaser_z, chaser_theta, target_x, target_y, target_z, target_theta, 
                                          chaser_x_dot, chaser_y_dot, chaser_z_dot, chaser_theta_dot] 
                    
                    Note: chaser_theta_dot can be returned as a zero (it is discarded before being run through the network)
                    """
                    
                                    
                    print('rc.W',rc.W)  # example to see the positions, or you can get the velocities as well...
                    if rc.id == target_id: # we've found the target
                        policy_input[4] =  rc.X[0] # target X [north] =   North
                        policy_input[5] = -rc.X[1] # targey Y [west]  = - East
                        policy_input[6] =  rc.X[2] # target Z [up]    =   Up
                        policy_input[7] =  np.unwrap([last_target_yaw, -rc.W[2]])[1] # target yaw  [counter-clockwise] = -yaw [clockwise]
                        last_target_yaw = policy_input[7]
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                    if rc.id == follower_id: # we've found the chaser (follower)
                        policy_input[0] =  rc.X[0] # chaser X [north] =   North
                        policy_input[1] = -rc.X[1] # chaser Y [west]  = - East
                        policy_input[2] =  rc.X[2] # chaser Z [up]    =   Up                        
                        policy_input[3] =  np.unwrap([last_chaser_yaw, -rc.W[2]])[1] # chaser yaw  [counter-clockwise] = -yaw [clockwise]
                        last_chaser_yaw = policy_input[3]
                        
                        policy_input[8]  =  rc.V[0] # chaser V_x [north] =   North
                        policy_input[9]  = -rc.V[1] # chaser V_y [west]  = - East
                        policy_input[10] =  rc.V[2] # chaser V_z [up]    =   Up
                        policy_input[11] =  0 # dummy entry on the chaser angular velocity because it is irrelevant and discarded
                        # Note: rc.X returns position; rc.V returns velocity; rc.W returns attitude
                    
                
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
        
                # Send velocity/acceleration command to aircraft!
                g.move_at_ned_vel( yaw=-deep_guidance[0])
                g.accelerate(deep_guidance[1:])
                print("Policy input: ", policy_input, "Deep guidance command: ", deep_guidance)
    


        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            g.set_nav_mode()
            g.shutdown()
            sleep(0.2)
            exit()


if __name__ == '__main__':
    main()

#EOF