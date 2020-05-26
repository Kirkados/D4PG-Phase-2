
"""
This script provides the environment for a quadrotor runway inspection simulation.

Three 4-dof quadrotors are tasked with inspecting a runway. They must learn to 
work together as a team. Each cell of the runway needs to be inspected by any of 
the three quadrotors. Rewards are only given when a new cell has been inspected.

Each quadrotor knows where the other ones are. They all use the same policy network.
They also all know the state of the runway and they all receive rewards when a 
new cell is explored.

Altitude is considered but heading is not considered.

All dynamic environments I create will have a standardized architecture. The
reason for this is I have one learning algorithm and many environments. All
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
    - animating the motion (via the render method):
        - Rendering is done all in one shot by passing the completed states
          from a trial to the render() method.

Outputs:
    Reward must be of shape ()
    State must be of shape (OBSERVATION_SIZE,)
    Done must be a bool

Inputs:
    Action input is of shape (ACTION_SIZE,)

Communication with agent:
    The agent communicates to the environment through two queues:
        agent_to_env: the agent passes actions or reset signals to the environment
        env_to_agent: the environment returns information to the agent

Reward system:
        - A reward of +1 is given for finding an unexplored runway element
        - Penaltys may be given for collisions or proportional to the distance 
          between the quadrotors.

State clarity:
    - TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
      TOTAL_STATE is returned from the environment to the agent.
      A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
      The TOTAL_STATE is passed to the animator below to animate the motion.
      The chaser and target state are contained in the environment. They are packaged up before being returned to the agent.
      The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.


Started May 19, 2020
@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import os
import signal
import multiprocessing
import queue
from scipy.integrate import odeint # Numerical integrator

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

class Environment:

    def __init__(self):
        ##################################
        ##### Environment Properties #####
        ##################################
        self.NUMBER_OF_QUADS                  = 5 # Number of quadrotors working together to complete the task
        self.BASE_STATE_SIZE                  = self.NUMBER_OF_QUADS * 6 # [my_x, my_y, my_z, my_Vx, my_Vy, my_Vz, other1_x, other1_y, other1_z, other1_Vx, other1_Vy, other1_Vz, other2_x, other2_y, other2_z
                                                   #  other2_Vx, other2_Vy, other2_Vz]  
        self.RUNWAY_WIDTH                     = 124 # [m]
        self.RUNWAY_LENGTH                    = 12.5 # [m]
        self.RUNWAY_WIDTH_ELEMENTS            = 3 # [elements]
        self.RUNWAY_LENGTH_ELEMENTS           = 4 # [elements]
        self.IRRELEVANT_STATES                = [] # indices of states who are irrelevant to the policy network
        self.ACTION_SIZE                      = 3 # [my_x_dot_dot, my_y_dot_dot, my_z_dot_dot]
        self.LOWER_ACTION_BOUND               = np.array([-2.0, -2.0, -2.0]) # [m/s^2, m/s^2, m/s^2]
        self.UPPER_ACTION_BOUND               = np.array([ 2.0,  2.0,  2.0]) # [m/s^2, m/s^2, m/s^2]
        self.LOWER_STATE_BOUND_PER_QUAD       = np.array([  0.,   0.,   0., -3., -3., -3.]) # [m, m, m, m/s, m/s, m/s]
        self.UPPER_STATE_BOUND_PER_QUAD       = np.array([100., 100., 100.,  3.,  3.,  3.]) # [m, m, m, m/s, m/s, m/s]
        self.NORMALIZE_STATE                  = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE                        = True # whether or not to RANDOMIZE the state & target location
        self.POSITION_RANDOMIZATION_AMOUNT    = np.array([10.0, 10.0, 3.0]) # [m, m, m]
        self.INITIAL_QUAD_POSITION            = np.array([10.0, 10.0, 0.0]) # [m, m, m,]     
        self.MIN_V                            = -200.
        self.MAX_V                            =  300.
        self.N_STEP_RETURN                    =   5
        self.DISCOUNT_FACTOR                  =   0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                         =   0.2 # [s]
        self.DYNAMICS_DELAY                   =   0 # [timesteps of delay] how many timesteps between when an action is commanded and when it is realized
        self.AUGMENT_STATE_WITH_ACTION_LENGTH =   0 # [timesteps] how many timesteps of previous actions should be included in the state. This helps with making good decisions among delayed dynamics.
        self.AUGMENT_STATE_WITH_STATE_LENGTH  =   0 # [timesteps] how many timesteps of previous states should be included in the state
        self.MAX_NUMBER_OF_TIMESTEPS          = 500 # per episode
        self.ADDITIONAL_VALUE_INFO            = False # whether or not to include additional reward and value distribution information on the animations
        self.TOP_DOWN_VIEW                    = False # Animation property

        # Test time properties
        self.TEST_ON_DYNAMICS                 = True # Whether or not to use full dynamics along with a PD controller at test time
        self.KINEMATIC_NOISE                  = False # Whether or not to apply noise to the kinematics in order to simulate a poor controller
        self.KINEMATIC_NOISE_SD               = [0.02, 0.02, 0.02, np.pi/100] # The standard deviation of the noise that is to be applied to each element in the state
        self.FORCE_NOISE_AT_TEST_TIME         = False # [Default -> False] Whether or not to force kinematic noise to be present at test time

        # PD Controller Gains
        self.KI                               = 10.0 # Integral gain for the integral-linear acceleration controller
        
        # Physical properties
        self.LENGTH                           = 0.3  # [m] side length
        self.MASS                             = 10   # [kg]
        self.INERTIA                          = 1/12*self.MASS*(self.LENGTH**2 + self.LENGTH**2) # 0.15 [kg m^2]
        
        # Target collision properties
        self.COLLISION_DISTANCE               = self.LENGTH # [m] how close chaser and target need to be before a penalty is applied
        self.COLLISION_PENALTY                = 15           # [rewards/second] penalty given for colliding with target  

        # Additional properties
        self.VELOCITY_LIMIT                   = 3 # [m/s] maximum allowable velocity, a hard cap is enforced if this velocity is exceeded. Note: Paparazzi must also supply a hard velocity cap
        self.ACCELERATION_PENALTY             = 0.0 # [factor] how much to penalize all acceleration commands
        self.MINIMUM_CAMERA_ALTITUDE          = 0 # [m] minimum altitude above the runway to get a reliable camera shot. If below this altitude, the runway element is not considered explored
        
        
        # Performing some calculations  
        self.RUNWAY_STATE_SIZE                = self.RUNWAY_WIDTH_ELEMENTS * self.RUNWAY_LENGTH_ELEMENTS # how big the runway "grid" is                                                   
        self.TOTAL_STATE_SIZE                 = self.BASE_STATE_SIZE + self.RUNWAY_STATE_SIZE
        self.LOWER_STATE_BOUND                = np.concatenate([np.tile(self.LOWER_STATE_BOUND_PER_QUAD, self.NUMBER_OF_QUADS), np.zeros(self.RUNWAY_STATE_SIZE)]) # lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND                = np.concatenate([np.tile(self.UPPER_STATE_BOUND_PER_QUAD, self.NUMBER_OF_QUADS),  np.ones(self.RUNWAY_STATE_SIZE)]) # upper bound for each element of TOTAL_STATE        
        self.OBSERVATION_SIZE                 = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy


    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)

    ######################################
    ##### Resettings the Environment #####
    ######################################
    def reset(self, use_dynamics, test_time):
        # This method resets the state and returns it
        """ NOTES:
               - if use_dynamics = True -> use dynamics
               - if test_time = True -> do not add "controller noise" to the kinematics
        """        

        # Logging whether it is test time for this episode
        self.test_time = test_time
        
        self.quad_positions = np.zeros([self.NUMBER_OF_QUADS, len(self.INITIAL_QUAD_POSITION)])

        # If we are randomizing the initial conditions and state
        if self.RANDOMIZE:
            # Randomizing initial state
            for i in range(self.NUMBER_OF_QUADS):
                self.quad_positions[i] = self.INITIAL_QUAD_POSITION + np.random.randn(len(self.POSITION_RANDOMIZATION_AMOUNT))*self.POSITION_RANDOMIZATION_AMOUNT

        else:
            # Constant initial state
            for i in range(self.NUMBER_OF_QUADS):
                self.quad_positions[i] = self.INITIAL_QUAD_POSITION

        # Quadrotors' initial velocity is not randomized
        self.quad_velocities = np.zeros([self.NUMBER_OF_QUADS, len(self.INITIAL_QUAD_POSITION)])
        
        # Initializing the previous velocity and control effort for the integral-acceleration controller
        self.previous_quad_velocities = np.zeros([self.NUMBER_OF_QUADS, len(self.INITIAL_QUAD_POSITION)])
        self.previous_linear_control_efforts = np.zeros([self.NUMBER_OF_QUADS, self.ACTION_SIZE])        
        
        if use_dynamics:            
            self.dynamics_flag = True # for this episode, dynamics will be used
        else:
            self.dynamics_flag = False # the default is to use kinematics

        # Resetting the time
        self.time = 0.
        
        # Resetting the runway state
        self.runway_state = np.zeros([self.RUNWAY_LENGTH_ELEMENTS, self.RUNWAY_WIDTH_ELEMENTS])
        self.previous_runway_value = 0
        
        # Resetting the action delay queue        
        if self.DYNAMICS_DELAY > 0:
            self.action_delay_queue = queue.Queue(maxsize = self.DYNAMICS_DELAY + 1)
            for i in range(self.DYNAMICS_DELAY):
                self.action_delay_queue.put(np.zeros([self.NUMBER_OF_QUADS, self.ACTION_SIZE]), False)

    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, actions):

        # Integrating forward one time step.
        # Returns initial condition on first row then next TIMESTEP on the next row
        #########################################
        ##### PROPAGATE KINEMATICS/DYNAMICS #####
        #########################################
        if self.dynamics_flag:
            ############################
            #### PROPAGATE DYNAMICS ####
            ############################

            # Next, calculate the control effort
            control_efforts = self.controller(actions)

            # Anything additional that needs to be sent to the dynamics integrator
            dynamics_parameters = [control_efforts, self.MASS, self.INERTIA, self.NUMBER_OF_QUADS, self.QUAD_POSITION_LENGTH]

            # Propagate the dynamics forward one timestep
            next_states = odeint(dynamics_equations_of_motion, np.concatenate([self.quad_positions, self.quad_velocities]), [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

            # Saving the new state
            for i in range(self.NUMBER_OF_QUADS):
                self.quad_positions[i]  = next_states[1,i*2*len(self.INITIAL_QUAD_POSITION):(i*2 + 1)*len(self.INITIAL_QUAD_POSITION)] # extract position
                self.quad_velocities[i] = next_states[1,(i*2 + 1)*len(self.INITIAL_QUAD_POSITION):(i + 1)*2*len(self.INITIAL_QUAD_POSITION)] # extract velocity
            #self.chaser_velocity[:-1] = np.clip(self.chaser_velocity[:-1], -self.VELOCITY_LIMIT, self.VELOCITY_LIMIT) # clipping the linear velocity to be within the limits
            
            print(next_states[1,:], self.quad_positions, self.quad_velocities)

        else:

            # Additional parameters to be passed to the kinematics
            kinematics_parameters = [actions, self.NUMBER_OF_QUADS, self.INITIAL_QUAD_POSITION]

            ###############################
            #### PROPAGATE KINEMATICS #####
            ###############################
            next_states = odeint(kinematics_equations_of_motion, np.concatenate([self.quad_positions, self.quad_velocities]), [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new state
            for i in range(self.NUMBER_OF_QUADS):
                self.quad_positions[i]  = next_states[1,i*2*len(self.INITIAL_QUAD_POSITION):(i*2 + 1)*len(self.INITIAL_QUAD_POSITION)] # extract position
                self.quad_velocities[i] = next_states[1,(i*2 + 1)*len(self.INITIAL_QUAD_POSITION):(i + 1)*2*len(self.INITIAL_QUAD_POSITION)] # extract velocity
                self.quad_velocities[i] = np.clip(self.quad_velocities[i], -self.VELOCITY_LIMIT, self.VELOCITY_LIMIT) # clipping the linear velocity to be within the limits

        # Done the differences between the kinematics and dynamics
        # Increment the timestep
        self.time += self.TIMESTEP
        
        # Update the state of the runway
        self.check_runway()

        # Calculating the reward for this state-action pair
        reward = self.reward_function(actions)

        # Check if this episode is done
        done = self.is_done()

        # Return the (reward, done)
        return reward, done

    def controller(self, actions):
        # This function calculates the control effort based on the state and the
        # desired acceleration (action)
        
        ###########################################################
        ### Position control (integral-acceleration controller) ###
        ###########################################################
        desired_linear_accelerations = actions
        
        current_velocities = self.chaser_velocities # [v_x, v_y, v_z]
        current_linear_acceleration = (current_velocities - self.previous_velocities)/self.TIMESTEP # Approximating the current acceleration [a_x, a_y, a_z]
        
        # Checking whether our velocity is too large AND the acceleration is trying to increase said velocity... in which case we set the desired_linear_acceleration to zero.
        desired_linear_accelerations[(np.abs(current_velocities) > self.VELOCITY_LIMIT) & (np.sign(desired_linear_accelerations) == np.sign(current_velocities))] = 0        
        
        # Calculating acceleration error
        linear_acceleration_error = desired_linear_accelerations - current_linear_acceleration
        
        # Integral-acceleration control
        linear_control_effort = self.previous_linear_control_effort + self.KI * linear_acceleration_error
        
        # Saving the current velocity for the next timetsep
        self.previous_velocities = current_velocities
        
        # Saving the current control effort for the next timestep
        self.previous_linear_control_effort = linear_control_effort

        return linear_control_effort

    def check_runway(self):
        # This method updates the runway state based off the current quadrotor positions
        """ The runway is 
        self.RUNWAY_WIDTH                     = 12.5 # [m]
        self.RUNWAY_LENGTH                    = 124 # [m]
        self.RUNWAY_WIDTH_ELEMENTS            = 6 # [elements]
        self.RUNWAY_LENGTH_ELEMENTS           = 16 # [elements]
        
        """
        each_runway_length_element = self.RUNWAY_LENGTH/self.RUNWAY_LENGTH_ELEMENTS
        each_runway_width_element  = self.RUNWAY_WIDTH/self.RUNWAY_WIDTH_ELEMENTS
        # Looping through each quad, determining which zone they've entered, and checking if that zone has been entered before

        # Which zones are the quads in?
        rows = np.floor(self.quad_positions[:,0]/each_runway_length_element)     
        rows = np.delete(rows, (rows < 0) | (rows >= self.RUNWAY_LENGTH_ELEMENTS) | (self.quad_positions[:,2] < self.MINIMUM_CAMERA_ALTITUDE))
        columns = np.floor(self.quad_positions[:,1]/each_runway_width_element)
        columns = np.delete(columns, (columns < 0) | (columns >= self.RUNWAY_WIDTH_ELEMENTS) | (self.quad_positions[:,2] < self.MINIMUM_CAMERA_ALTITUDE))
        
        # If appropriate, mark the visited tiles as explored
        self.runway_state[rows,columns] = 1
        

    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action
        
        # Initializing the rewards to zero for all quads
        rewards = np.zeros(self.NUMBER_OF_QUADS)
        
        # Give rewards according to the change in runway state. A newly explored tile will yield a reward of +1
        rewards += np.sum(self.runway_state) - self.previous_runway_value
        
        # Storing the current runway state for the next timestep
        self.previous_runway_value = np.sum(self.runway_state)

        # Penalizing acceleration commands (to encourage fuel efficiency)
        rewards -= np.sum(self.ACCELERATION_PENALTY*np.abs(action), axis = 1)

        return rewards

    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """
        # Initializing
        done = False
        
        # If we've explored the entire runway
        if np.sum(self.runway_state) == self.RUNWAY_STATE_SIZE:
            done = True

        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            done = True

        return done


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent
    

    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Loop until the process is terminated
        while True:
            # Blocks until the agent passes us an action
            actions, *test_time = self.agent_to_env.get()        

            if np.any(type(actions) == bool):
                # The signal to reset the environment was received
                self.reset(actions, test_time[0])
                
                # Return the TOTAL_STATE           
                self.env_to_agent.put((self.quad_positions, self.quad_velocities, self.runway_state))

            else:
                # Delay the action by DYNAMICS_DELAY timesteps. The environment accumulates the action delay--the agent still thinks the sent action was used.
                if self.DYNAMICS_DELAY > 0:
                    self.action_delay_queue.put(actions,False) # puts the current action to the bottom of the stack
                    actions = self.action_delay_queue.get(False) # grabs the delayed action and treats it as truth.                
                
                ################################
                ##### Step the environment #####
                ################################                
                rewards, done = self.step(actions)

                # Return (TOTAL_STATE, reward, done, guidance_position)
                self.env_to_agent.put((self.quad_positions, self.quad_velocities, self.runway_state, rewards, done))

###################################################################
##### Generating kinematics equations representing the motion #####
###################################################################
def kinematics_equations_of_motion(state, t, parameters):
    """ 
    Returns the first derivative of the state
    The state is [position, velocity]; its derivative is [velocity, acceleration]
    """
    
    # Unpacking the action from the parameters
    actions = parameters[0]
    NUMBER_OF_QUADS = parameters[1]
    QUAD_POSITION_LENGTH = parameters[2]
    
    # state = quad_positions, quad_velocities concatenated
    #quad_positions  = state[:NUMBER_OF_QUADS*QUAD_POSITION_LENGTH]
    quad_velocities = state[NUMBER_OF_QUADS*QUAD_POSITION_LENGTH:]

    # Flattening the accelerations into a column
    accelerations = actions.reshape(-1) # [x_dot_dot, y_dot_dot, z_dot_dot, x_dot_dot, y_dot_dot....]

    # Building the derivative matrix.
    derivatives = np.concatenate([quad_velocities, accelerations]) #.squeeze()?

    return derivatives


#####################################################################
##### Generating the dynamics equations representing the motion #####
#####################################################################
def dynamics_equations_of_motion(state, t, parameters):
    """ 
    Returns the first derivative of the state
    The state is [position, velocity]; its derivative is [velocity, acceleration]
    """
    # Unpacking the parameters
    control_efforts, mass, inertia, NUMBER_OF_QUADS, QUAD_POSITION_LENGTH = parameters

    # Unpacking the state
    #quad_positions  = state[:NUMBER_OF_QUADS*QUAD_POSITION_LENGTH]
    quad_velocities = state[NUMBER_OF_QUADS*QUAD_POSITION_LENGTH:]
    
    # Calculating accelerations = F/m
    accelerations = control_efforts.reshape(-1)/mass

    # Building derivatives array
    derivatives = np.concatenate([quad_velocities, accelerations]) #.squeeze()?

    return derivatives


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, instantaneous_reward_log, cumulative_reward_log, critic_distributions, target_critic_distributions, projected_target_distribution, bins, loss_log, episode_number, filename, save_directory):

    # Load in a temporary environment, used to grab the physical parameters
    temp_env = Environment()

    # Checking if we want the additional reward and value distribution information
    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    # Unpacking state
    chaser_x, chaser_y, chaser_z, chaser_theta = states[:,0], states[:,1], states[:,2], states[:,3]
    
    target_x, target_y, target_z, target_theta = states[:,4], states[:,5], states[:,6], states[:,7]

    # Extracting physical properties
    length = temp_env.LENGTH

    ### Calculating spacecraft corner locations through time ###
    
    # Corner locations in body frame    
    chaser_body_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                                [[-1],[-1],[1]],
                                                [[-1],[-1],[-1]],
                                                [[1],[-1],[-1]],
                                                [[-1],[-1],[-1]],
                                                [[-1],[1],[-1]],
                                                [[1],[1],[-1]],
                                                [[-1],[1],[-1]],
                                                [[-1],[1],[1]],
                                                [[1],[1],[1]],
                                                [[-1],[1],[1]],
                                                [[-1],[-1],[1]]]).squeeze().T
    
    chaser_front_face_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                                       [[1],[1],[1]],
                                                       [[1],[1],[-1]],
                                                       [[1],[-1],[-1]],
                                                       [[1],[-1],[1]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(chaser_theta),       -np.sin(chaser_theta),        np.zeros(len(chaser_theta))],
                                 [np.sin(chaser_theta),        np.cos(chaser_theta),        np.zeros(len(chaser_theta))],
                                 [np.zeros(len(chaser_theta)), np.zeros(len(chaser_theta)), np.ones(len(chaser_theta))]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 3, 3]
    
    # Rotating body frame coordinates to inertial frame
    chaser_body_inertial       = np.matmul(C_Ib, chaser_body_body_frame)       + np.array([chaser_x, chaser_y, chaser_z]).T.reshape([-1,3,1])
    chaser_front_face_inertial = np.matmul(C_Ib, chaser_front_face_body_frame) + np.array([chaser_x, chaser_y, chaser_z]).T.reshape([-1,3,1])

    ### Calculating target spacecraft corner locations through time ###
    
    # Corner locations in body frame    
    target_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                           [[-1],[-1],[1]],
                                           [[-1],[-1],[-1]],
                                           [[1],[-1],[-1]],
                                           [[-1],[-1],[-1]],
                                           [[-1],[1],[-1]],
                                           [[1],[1],[-1]],
                                           [[-1],[1],[-1]],
                                           [[-1],[1],[1]],
                                           [[1],[1],[1]],
                                           [[-1],[1],[1]],
                                           [[-1],[-1],[1]]]).squeeze().T
        
    target_front_face_body_frame = length/2.*np.array([[[1],[-1],[1]],
                                                       [[1],[1],[1]],
                                                       [[1],[1],[-1]],
                                                       [[1],[-1],[-1]],
                                                       [[1],[-1],[1]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(target_theta),       -np.sin(target_theta),        np.zeros(len(target_theta))],
                                 [np.sin(target_theta),        np.cos(target_theta),        np.zeros(len(target_theta))],
                                 [np.zeros(len(target_theta)), np.zeros(len(target_theta)), np.ones(len(target_theta))]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 3, 3]
    target_body_inertial = np.matmul(C_Ib, target_body_frame)+ np.array([target_x, target_y, target_z]).T.reshape([-1,3,1])
    target_front_face_inertial = np.matmul(C_Ib, target_front_face_body_frame) + np.array([target_x, target_y, target_z]).T.reshape([-1,3,1])

    # Generating figure window
    figure = plt.figure(constrained_layout = True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows = 2, ncols = 3, figure = figure)
        subfig1 = figure.add_subplot(grid_spec[0,0], projection = '3d', aspect = 'equal', autoscale_on = False, xlim3d = (-5, 5), ylim3d = (-5, 5), zlim3d = (0, 10), xlabel = 'X (m)', ylabel = 'Y (m)', zlabel = 'Z (m)')
        subfig2 = figure.add_subplot(grid_spec[0,1], xlim = (np.min([np.min(instantaneous_reward_log), 0]) - (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02, np.max([np.max(instantaneous_reward_log), 0]) + (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02), ylim = (-0.5, 0.5))
        subfig3 = figure.add_subplot(grid_spec[0,2], xlim = (np.min(loss_log)-0.01, np.max(loss_log)+0.01), ylim = (-0.5, 0.5))
        subfig4 = figure.add_subplot(grid_spec[1,0], ylim = (0, 1.02))
        subfig5 = figure.add_subplot(grid_spec[1,1], ylim = (0, 1.02))
        subfig6 = figure.add_subplot(grid_spec[1,2], ylim = (0, 1.02))

        # Setting titles
        subfig1.set_xlabel("X (m)",    fontdict = {'fontsize': 8})
        subfig1.set_ylabel("Y (m)",    fontdict = {'fontsize': 8})
        subfig2.set_title("Timestep Reward",    fontdict = {'fontsize': 8})
        subfig3.set_title("Current loss",       fontdict = {'fontsize': 8})
        subfig4.set_title("Q-dist",             fontdict = {'fontsize': 8})
        subfig5.set_title("Target Q-dist",      fontdict = {'fontsize': 8})
        subfig6.set_title("Bellman projection", fontdict = {'fontsize': 8})

        # Changing around the axes
        subfig1.tick_params(labelsize = 8)
        subfig2.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig3.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig4.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig5.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig6.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 8)

        # Adding the grid
        subfig4.grid(True)
        subfig5.grid(True)
        subfig6.grid(True)

        # Setting appropriate axes ticks
        subfig2.set_xticks([np.min(instantaneous_reward_log), 0, np.max(instantaneous_reward_log)] if np.sign(np.min(instantaneous_reward_log)) != np.sign(np.max(instantaneous_reward_log)) else [np.min(instantaneous_reward_log), np.max(instantaneous_reward_log)])
        subfig3.set_xticks([np.min(loss_log), np.max(loss_log)])
        subfig4.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig4.tick_params(axis = 'x', labelrotation = -90)
        subfig4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig5.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig5.tick_params(axis = 'x', labelrotation = -90)
        subfig5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig6.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig6.tick_params(axis = 'x', labelrotation = -90)
        subfig6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])

    else:
        subfig1 = figure.add_subplot(1, 1, 1, projection = '3d', aspect = 'equal', autoscale_on = False, xlim3d = (-5, 5), ylim3d = (-5, 5), zlim3d = (0, 10), xlabel = 'X (m)', ylabel = 'Y (m)', zlabel = 'Z (m)')
    
    # Setting the proper view
    if temp_env.TOP_DOWN_VIEW:
        subfig1.view_init(-90,0)
    else:
        subfig1.view_init(25, 190)        

    # Defining plotting objects that change each frame
    chaser_body,       = subfig1.plot([], [], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed
    chaser_front_face, = subfig1.plot([], [], [], color = 'k', linestyle = '-', linewidth = 2) # Note, the comma is needed
    target_body,       = subfig1.plot([], [], [], color = 'g', linestyle = '-', linewidth = 2)
    target_front_face, = subfig1.plot([], [], [], color = 'k', linestyle = '-', linewidth = 2)
    chaser_body_dot    = subfig1.scatter(0., 0., 0., color = 'r', s = 0.1)

    if extra_information:
        reward_bar           = subfig2.barh(y = 0, height = 0.2, width = 0)
        loss_bar             = subfig3.barh(y = 0, height = 0.2, width = 0)
        q_dist_bar           = subfig4.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        target_q_dist_bar    = subfig5.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        projected_q_dist_bar = subfig6.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        time_text            = subfig1.text2D(x = 0.2, y = 0.91, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text          = subfig1.text2D(x = 0.0,  y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)
    else:        
        time_text    = subfig1.text2D(x = 0.1, y = 0.9, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text  = subfig1.text2D(x = 0.62, y = 0.9, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text = subfig1.text2D(x = 0.4, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text.set_text('Episode ' + str(episode_number))

    # Function called repeatedly to draw each frame
    def render_one_frame(frame, *fargs):
        temp_env = fargs[0] # Extract environment from passed args

        # Draw the chaser body
        chaser_body.set_data(chaser_body_inertial[frame,0,:], chaser_body_inertial[frame,1,:])
        chaser_body.set_3d_properties(chaser_body_inertial[frame,2,:])

        # Draw the front face of the chaser body in a different colour
        chaser_front_face.set_data(chaser_front_face_inertial[frame,0,:], chaser_front_face_inertial[frame,1,:])
        chaser_front_face.set_3d_properties(chaser_front_face_inertial[frame,2,:])

        # Draw the target body
        target_body.set_data(target_body_inertial[frame,0,:], target_body_inertial[frame,1,:])
        target_body.set_3d_properties(target_body_inertial[frame,2,:])

        # Draw the front face of the target body in a different colour
        target_front_face.set_data(target_front_face_inertial[frame,0,:], target_front_face_inertial[frame,1,:])
        target_front_face.set_3d_properties(target_front_face_inertial[frame,2,:])

        # Drawing a dot in the centre of the chaser
        chaser_body_dot._offsets3d = ([chaser_x[frame]],[chaser_y[frame]],[chaser_z[frame]])

        # Update the time text
        time_text.set_text('Time = %.1f s' %(frame*temp_env.TIMESTEP))

#       # Update the reward text
        reward_text.set_text('Total reward = %.1f' %cumulative_reward_log[frame])
#
        if extra_information:
            # Updating the instantaneous reward bar graph
            reward_bar[0].set_width(instantaneous_reward_log[frame])
            # And colouring it appropriately
            if instantaneous_reward_log[frame] < 0:
                reward_bar[0].set_color('r')
            else:
                reward_bar[0].set_color('g')

            # Updating the loss bar graph
            loss_bar[0].set_width(loss_log[frame])

            # Updating the q-distribution plot
            for this_bar, new_value in zip(q_dist_bar, critic_distributions[frame,:]):
                this_bar.set_height(new_value)

            # Updating the target q-distribution plot
            for this_bar, new_value in zip(target_q_dist_bar, target_critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            # Updating the projected target q-distribution plot
            for this_bar, new_value in zip(projected_q_dist_bar, projected_target_distribution[frame, :]):
                this_bar.set_height(new_value)
#
        # Since blit = True, must return everything that has changed at this frame
        return chaser_body_dot, time_text, chaser_body, chaser_front_face, target_body, target_front_face 

    # Generate the animation!
    fargs = [temp_env] # bundling additional arguments
    animator = animation.FuncAnimation(figure, render_one_frame, frames = np.linspace(0, len(states)-1, len(states)).astype(int),
                                       blit = False, fargs = fargs)

    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """

    # Save the animation!
    try:
        # Save it to the working directory [have to], then move it to the proper folder
        animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
        # Make directory if it doesn't already exist
        os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
        # Move animation to the proper directory
        os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
    except:
        print("Skipping animation for episode %i due to an error" %episode_number)
        # Try to delete the partially completed video file
        try:
            os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
        except:
            pass

    del temp_env
    plt.close(figure)