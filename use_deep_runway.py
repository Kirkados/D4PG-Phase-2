#!/usr/bin/env python

from __future__ import print_function


from time import sleep
import numpy as np
import queue
from shapely.geometry import LineString, Polygon

# Deep guidance stuff
import tensorflow as tf
import time

from settings import Settings
from build_neural_networks import BuildActorNetwork

# Paparazzi guidance api
from guidance_common import Guidance

def make_C_bI(angle):
    
    C_bI = np.array([[ np.cos(angle), np.sin(angle), 0],
                     [-np.sin(angle), np.cos(angle), 0],
                     [             0,             0, 1]]) # [3, 3]        
    return C_bI

def make_C_bI_22(angle):
    
    C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
    return C_bI


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guided mode example")
    parser.add_argument('-ids','--quad_ids', nargs='+', help='<Required> IDs of all quads used', required=True)
    parser.add_argument("-f", "--filename", dest='log_filename', default='log_runway_000', type=str, help="Log file name")
    parser.add_argument("-L", "--new_length", dest='new_length', default=Settings.RUNWAY_LENGTH, type=float, help="Override the 124 m runway length")
    parser.add_argument("-W", "--new_width", dest='new_width', default=Settings.RUNWAY_WIDTH, type=float, help="Override the 12.5 m runway width")
    parser.add_argument("-alt", "--altitude", dest='altitude', default=2, type=float, help="Override the 2 m first quad altitude")
    parser.add_argument("-no_avg", "--dont_average_output", dest="dont_average_output", action="store_true")
    parser.add_argument("-fail", "--failure_times", nargs='+', dest="failure_times",default=[9999.9])
    args = parser.parse_args()

    failure_times = list(map(float, args.failure_times))
  
    
    if Settings.INDOORS:
        runway_angle_wrt_north = 0
    else:
        runway_angle_wrt_north = (156.485-90)*np.pi/180 #[rad]
    C_bI = make_C_bI(runway_angle_wrt_north) # [3x3] rotation matrix from North (I) to body (tilted runway)
    C_bI_22 = make_C_bI_22(runway_angle_wrt_north) # [2x2] rotation matrix from North (I) to body (tilted runway)
    
    if args.new_length != Settings.RUNWAY_LENGTH:
        print("\nOverwriting %.1f m runway length with user-defined %.1f m runway length." %(Settings.RUNWAY_LENGTH,args.new_length))
    runway_length = args.new_length
    if args.new_width != Settings.RUNWAY_WIDTH:
        print("\nOverwriting %.1f m runway width with user-defined %.1f m runway width." %(Settings.RUNWAY_WIDTH,args.new_width))
    runway_width = args.new_width
    
    first_quad_altitude = args.altitude
    
    # Communication delay length in timesteps
    COMMUNICATION_DELAY_LENGTH = 0 # timesteps
    if COMMUNICATION_DELAY_LENGTH > 0:
        print("\nA simulated communication delay of %.1f seconds is used" %(COMMUNICATION_DELAY_LENGTH*Settings.TIMESTEP))

    interface = None
    not_done = True
    
    failure_acknowledged = np.tile(False,len(failure_times))
    
    average_deep_guidance_NED = np.zeros([Settings.NUMBER_OF_QUADS, Settings.ACTION_SIZE])
    
    # converting this input from a list of strings to a list of ints
    all_ids = list(map(int, args.quad_ids))
    
    log_filename = args.log_filename
    max_duration = 100000
    log_placeholder = np.zeros((max_duration, 3*Settings.NUMBER_OF_QUADS + 1 + Settings.TOTAL_STATE_SIZE + 6))
    log_counter = 0 # for log increment
    
    # Flag to not average the guidance output
    dont_average_output = args.dont_average_output
    if dont_average_output:
        print("\n\nDeep guidance output is NOT averaged\n\n")
    else:
        print("\n\nDeep guidance output is averaged\n\n")
    
    timestep = Settings.TIMESTEP
    
    
    # Generate Polygons for runway tiles
    # The size of each runway grid element
    each_runway_length_element = Settings.RUNWAY_LENGTH/Settings.RUNWAY_LENGTH_ELEMENTS
    each_runway_width_element  = Settings.RUNWAY_WIDTH/Settings.RUNWAY_WIDTH_ELEMENTS
    tile_polygons = []
    for i in range(Settings.RUNWAY_LENGTH_ELEMENTS):
        this_row = []
        for j in range(Settings.RUNWAY_WIDTH_ELEMENTS):
            # make the polygon
            this_row.append(Polygon([[each_runway_length_element*i     - Settings.RUNWAY_LENGTH/2, each_runway_width_element*j     - Settings.RUNWAY_WIDTH/2],
                                     [each_runway_length_element*(i+1) - Settings.RUNWAY_LENGTH/2, each_runway_width_element*j     - Settings.RUNWAY_WIDTH/2],
                                     [each_runway_length_element*(i+1) - Settings.RUNWAY_LENGTH/2, each_runway_width_element*(j+1) - Settings.RUNWAY_WIDTH/2],
                                     [each_runway_length_element*i     - Settings.RUNWAY_LENGTH/2, each_runway_width_element*(j+1) - Settings.RUNWAY_WIDTH/2]]))
            
        tile_polygons.append(this_row)
    
    
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
            g = Guidance(interface=interface, quad_ids = all_ids)
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
            
            # Initializing
            runway_state = np.zeros([Settings.RUNWAY_LENGTH_ELEMENTS, Settings.RUNWAY_WIDTH_ELEMENTS])
            # A list of all the tiles that rewards should not be returned for
            if Settings.RUNWAY_SHAPE == 'R':
                blank_tiles = []
                print("Rectangular-shaped runway is used!")
                
            elif Settings.RUNWAY_SHAPE == 'L':
                blank_tiles = list(([2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1]))
                print("L-shaped runway is used!")
                
            elif Settings.RUNWAY_SHAPE == 'C':                
                blank_tiles = list(([2,0],[3,0],[4,0],[5,0],[2,1],[3,1],[4,1],[5,1]))
                print("C-shaped runway is used!")
            
            # Assign tiles we don't care about to 1 (meaning they won't give any reward)
            for i in range(len(blank_tiles)):
                runway_state[blank_tiles[i][0],blank_tiles[i][1]] = 1
            
            last_runway_state = np.copy(runway_state)
            desired_altitudes = np.linspace(first_quad_altitude+Settings.NUMBER_OF_QUADS*2, first_quad_altitude, Settings.NUMBER_OF_QUADS, endpoint = False)
                
            while not_done:
                # TODO: make better frequency managing
                sleep(timestep)
                
                # Initializing quadrotor positions and velocities
                quad_positions = np.zeros([Settings.NUMBER_OF_QUADS, 3]) 
                quad_velocities = np.zeros([Settings.NUMBER_OF_QUADS, 3])
                
                quad_number_not_id = 0
                for rc in g.rotorcrafts:
                    
                    rc.timeout = rc.timeout + timestep                    
                    
                    """ policy_input is: [chaser_x, chaser_y, chaser_z, target_x, target_y, target_z, target_theta, 
                                          chaser_x_dot, chaser_y_dot, chaser_z_dot, (optional past action data)] 
                    """

                    # Extracting position
                    quad_positions[ quad_number_not_id, 0] =  rc.X[0]
                    quad_positions[ quad_number_not_id, 1] = -rc.X[1]
                    quad_positions[ quad_number_not_id, 2] =  rc.X[2]
                    
                    # Rotating from Inertial frame (NED frame) into body frame (runway frame)
                    quad_positions[ quad_number_not_id, :] = np.matmul(C_bI, quad_positions[ quad_number_not_id, :])
                    
                    # Scale position after rotating
                    quad_positions[ quad_number_not_id, 0] = quad_positions[ quad_number_not_id, 0]*Settings.RUNWAY_LENGTH/runway_length # scaling to the new runway length
                    quad_positions[ quad_number_not_id, 1] = quad_positions[ quad_number_not_id, 1]*Settings.RUNWAY_WIDTH/runway_width # scaling to the new runway wodth
                    
                    # Extracting velocity
                    quad_velocities[quad_number_not_id, 0] =  rc.V[0]#*Settings.RUNWAY_LENGTH/runway_length
                    quad_velocities[quad_number_not_id, 1] = -rc.V[1]#*Settings.RUNWAY_WIDTH/runway_width
                    quad_velocities[quad_number_not_id, 2] =  rc.V[2]
                    
                    # Rotating from Inertial frame (NED frame) into body frame (runway frame)
                    quad_velocities[ quad_number_not_id, :] = np.matmul(C_bI, quad_velocities[ quad_number_not_id, :])
                    
                    # Scale velocities after rotating
                    quad_velocities[quad_number_not_id, 0] = quad_velocities[quad_number_not_id, 0]*Settings.RUNWAY_LENGTH/runway_length
                    quad_velocities[quad_number_not_id, 1] = quad_velocities[quad_number_not_id, 1]*Settings.RUNWAY_WIDTH/runway_width

                    quad_number_not_id += 1
                
                
                
                # Resetting the action delay queue 
                if total_time == 0.0:                        
                    if COMMUNICATION_DELAY_LENGTH > 0:
                        communication_delay_queue = queue.Queue(maxsize = COMMUNICATION_DELAY_LENGTH + 1)
                        # Fill it with zeros initially
                        for i in range(COMMUNICATION_DELAY_LENGTH):
                            communication_delay_queue.put([quad_positions, quad_velocities], False)
                    
                    # Resetting the initial previous position to be the first position
                    previous_quad_positions = quad_positions
                
                
                # Checking if a quadrotor has failed
                for i in range(len(failure_times)):
                    if (total_time > failure_times[i]) and (not failure_acknowledged[i]):
                        if total_time == failure_times[i]:
                            print("\n\nQuad %i has failed!\n\n"%i)
                        # Force the position and velocity to their 'failed' states
                        quad_positions[i,:]  = Settings.LOWER_STATE_BOUND[:3]
                        previous_quad_positions[i,:] = quad_positions[i,:]
                        quad_velocities[i,:] = np.zeros([3])
                                    
                if COMMUNICATION_DELAY_LENGTH > 0:
                    communication_delay_queue.put([quad_positions, quad_velocities], False) # puts the current position and velocity to the bottom of the stack
                    delayed_quad_positions, delayed_quad_velocities = communication_delay_queue.get(False) # grabs the delayed position and velocity.   
                                
                ##############################################################
                ### Check the runway for new tiles that have been explored ###
                ##############################################################
                # Generate quadrotor LineStrings
                for i in range(Settings.NUMBER_OF_QUADS):
                    quad_line = LineString([quad_positions[i,:-1], previous_quad_positions[i,:-1]])
                    
                    for j in range(Settings.RUNWAY_LENGTH_ELEMENTS):
                        for k in range(Settings.RUNWAY_WIDTH_ELEMENTS):                    
                            # If this element has already been explored, skip it
                            if runway_state[j,k] == 0 and quad_line.intersects(tile_polygons[j][k]):
                                runway_state[j,k] = 1
                                #print("Quad %i traced the line %s and explored runway element length = %i, width = %i with coordinates %s" %(i,list(quad_line.coords),j,k,tile_polygons[j][k].bounds))
                    
                # Storing current quad positions for the next timestep
                previous_quad_positions = quad_positions
                
                # Print if a new tile has been explored
                if np.any(last_runway_state != runway_state):
                    print("Runway elements discovered %i/%i" %(np.sum(runway_state), Settings.RUNWAY_LENGTH_ELEMENTS*Settings.RUNWAY_WIDTH_ELEMENTS))
                    
                    # Draw a new runway
                    print(np.flip(runway_state))
                
                if np.all(runway_state) == 1:
                    print("Explored the entire runway in %.2f seconds--Congratualtions! Quitting deep guidance" %(time.time()-start_time))
                    not_done = False
                
                total_states = []
                # Building NUMBER_OF_QUADS states
                for j in range(Settings.NUMBER_OF_QUADS):
                    # Start state with your own 
                    this_quads_state = np.concatenate([quad_positions[j,:], quad_velocities[j,:]])               
                    # Add in the others' states, starting with the next quad and finishing with the previous quad
                    for k in range(j + 1, Settings.NUMBER_OF_QUADS + j):
                        if COMMUNICATION_DELAY_LENGTH > 0:
                            this_quads_state = np.concatenate([this_quads_state, delayed_quad_positions[k % Settings.NUMBER_OF_QUADS,:], delayed_quad_velocities[k % Settings.NUMBER_OF_QUADS,:]])
                        else:
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
                
                total_state_to_log = total_states[0,:]

                # Normalize the state
                if Settings.NORMALIZE_STATE:
                    total_states = (total_states - Settings.STATE_MEAN)/Settings.STATE_HALF_RANGE

                # Discarding irrelevant states
                observations = np.delete(total_states, Settings.IRRELEVANT_STATES, axis = 1)

                # Run processed state through the policy
                deep_guidance = sess.run(actor.action_scaled, feed_dict={state_placeholder:observations}) # deep guidance = [ chaser_x_acceleration [runway north], chaser_y_acceleration [runway west], chaser_z_acceleration [up] ]
                
                # Adding the action taken to the past_action log
                if Settings.AUGMENT_STATE_WITH_ACTION_LENGTH > 0:
                    past_actions.put(deep_guidance)

                # Limit guidance commands if velocity is too high!
                # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
                for j in range(Settings.NUMBER_OF_QUADS):
                    #print(quad_velocities[j,0:2], (np.abs(quad_velocities[j,0:2]) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance[j,:]) == np.sign(quad_velocities[j,0:2])), "Unsaturated: ", deep_guidance, end=' ')
                    deep_guidance[j,(np.abs(quad_velocities[j,0:2]) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance[j,:]) == np.sign(quad_velocities[j,0:2]))] = -2*np.sign(deep_guidance[j,(np.abs(quad_velocities[j,0:2]) > Settings.VELOCITY_LIMIT) & (np.sign(deep_guidance[j,:]) == np.sign(quad_velocities[j,0:2]))])
                    #print("Saturated: ", deep_guidance, end =' ')
                    
        
                average_deep_guidance = (last_deep_guidance + deep_guidance)/2.0
                last_deep_guidance = deep_guidance
                last_runway_state = np.copy(runway_state)
                
                # Get each quad to accelerate appropriately
                for j in range(Settings.NUMBER_OF_QUADS):
                    
                    # Checking if a quadrotor has failed
                    for i in range(len(failure_times)):                            
                        if (total_time > failure_times[i]) and (i==j):
                            if np.abs(total_time - failure_times[i]) < 0.5:
                                print("\n\nSimulated failure of quad %i\n\n"%(all_ids[i]))
                            skip_this_one = True
                            break
                        else:
                            skip_this_one = False
                    if skip_this_one:
                        continue
                    
                    # Each quad is assigned a different altitude to remain at.
                    desired_altitude = desired_altitudes[j]

                    # Rotate desired deep guidance command from body frame (runway frame) frame to inertial frame (NED frame)
                    #print("Unrotated average: ", average_deep_guidance)
                    average_deep_guidance_NED[j,:] = np.matmul(C_bI_22.T, average_deep_guidance[j,:])
                    #print("Rotated average: ", average_deep_guidance)
                    
                    
                    if dont_average_output:
                        g.accelerate(north = deep_guidance[j,0], east = -deep_guidance[j,1], down = desired_altitude, quad_id = g.ids[j])
                        print("The no_avg setting is broken")
                        raise SystemExit
                    else:
                        g.accelerate(north = average_deep_guidance_NED[j,0], east = -average_deep_guidance_NED[j,1], down = desired_altitude, quad_id = g.ids[j]) # Averaged        
                
                total_time = total_time + timestep
                # Log all input and outputs:
                t = time.time()-start_time
                log_placeholder[log_counter,0] = t
                log_placeholder[log_counter,1:3*Settings.NUMBER_OF_QUADS + 1] = np.concatenate([average_deep_guidance.reshape(-1), desired_altitudes.reshape(-1)])
                # log_placeholder[i,5:8] = deep_guidance_xf, deep_guidance_yf, deep_guidance_zf
                log_placeholder[log_counter,3*Settings.NUMBER_OF_QUADS + 1:3*Settings.NUMBER_OF_QUADS + 1 + Settings.TOTAL_STATE_SIZE + 6] = total_state_to_log
                log_counter += 1
    
            # If we ended gracefully
            exit()
        
        # If we ended forcefully
        except (KeyboardInterrupt, SystemExit): 
            print('Shutting down...')
            g.shutdown()
            sleep(0.2)
            print("Saving file as %s..." %(log_filename+"_L"+str(runway_length)+"_W"+str(runway_width)+".txt"))
            with open(log_filename+"_L"+str(runway_length)+"_W"+str(runway_width)+".txt", 'wb') as f:
                np.save(f, log_placeholder[:log_counter])
            print("Done!")
            exit()


if __name__ == '__main__':
    main()

#EOF