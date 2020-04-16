#!/usr/bin/env python

from __future__ import print_function

import sys
from os import path, getenv

# if PAPARAZZI_SRC or PAPARAZZI_HOME not set, then assume the tree containing this
# file is a reasonable substitute
PPRZ_HOME = getenv("PAPARAZZI_HOME", path.normpath(path.join(path.dirname(path.abspath(__file__)), '../../../../')))
PPRZ_SRC = getenv("PAPARAZZI_SRC", path.normpath(path.join(path.dirname(path.abspath(__file__)), '../../../../')))
sys.path.append(PPRZ_SRC + "/sw/lib/python")
sys.path.append(PPRZ_HOME + "/var/lib/python") # pprzlink

from pprzlink.ivy import IvyMessagesInterface
from pprzlink.message import PprzMessage
from settings_xml_parse import PaparazziACSettings

from math import radians
from time import sleep
import numpy as np

# Deep guidance stuff
import tensorflow as tf
import time

from settings import Settings
from build_neural_networks import BuildActorNetwork


class Rotorcraft:
    def __init__(self, ac_id):
        self.initialized = False
        self.id = ac_id
        self.X = np.zeros(3) # Position
        self.V = np.zeros(3) # Velocity
        self.W = np.zeros(3) # Angles
        self.timeout = 0

class Guidance(object):
    def __init__(self, verbose=False, interface=None, target_id=None, follower_id=None):
        self.verbose = verbose
        self.step = 0.1 # 10Hz
        self._interface = interface
        self.auto2_index = None
        self.target_id = target_id
        self.follower_id = follower_id
        self.ac_id = self.follower_id
        self.ids = [target_id, follower_id]
        self.ap_mode = None
        self.rotorcrafts = [Rotorcraft(i) for i in self.ids]

        try:
            settings = PaparazziACSettings(self.ac_id) # target and follower should be checked, FIX ME !
        except Exception as e:
            print(e)
            return
        try:
            self.ap_mode = settings.name_lookup['mode'] # try classic name
        except Exception as e:
            try:
                self.ap_mode = settings.name_lookup['ap'] # in case it is a generated autopilot
            except Exception as e:
                print(e)
                print("ap_mode setting not found, mode change not possible.")
        self._interface = IvyMessagesInterface("Deep Guidance is on the way...")

        # bind to INS message
        def ins_cb(ac_id, msg):
            if ac_id in self.ids and msg.name == "INS":
                rc = self.rotorcrafts[self.ids.index(ac_id)]
                i2p = 1. / 2**8     # integer to position
                i2v = 1. / 2**19    # integer to velocity
                rc.X[0] = float(msg['ins_x']) * i2p
                rc.X[1] = float(msg['ins_y']) * i2p
                rc.X[2] = float(msg['ins_z']) * i2p
                rc.V[0] = float(msg['ins_xd']) * i2v
                rc.V[1] = float(msg['ins_yd']) * i2v
                rc.V[2] = float(msg['ins_zd']) * i2v
                rc.timeout = 0
                rc.initialized = True
        # self._interface.subscribe(ins_cb, PprzMessage("telemetry", "INS"))

        def rotorcraft_fp_cb(ac_id, msg):
            if ac_id in self.ids and msg.name == "ROTORCRAFT_FP":
                rc = self.rotorcrafts[self.ids.index(ac_id)]
                i2p = 1. / 2**8     # integer to position
                i2v = 1. / 2**19    # integer to velocity
                i2w = 1. / 2**12     # integer to angle
                rc.X[0] = float(msg['north']) * i2p
                rc.X[1] = float(msg['east']) * i2p
                rc.X[2] = float(msg['up']) * i2p
                rc.V[0] = float(msg['vnorth']) * i2v
                rc.V[1] = float(msg['veast']) * i2v
                rc.V[2] = float(msg['vup']) * i2v
                rc.W[2] = float(msg['psi']) * i2w
                rc.timeout = 0
                rc.initialized = True
        self._interface.subscribe(rotorcraft_fp_cb, PprzMessage("telemetry", "ROTORCRAFT_FP"))
    # <message name="ROTORCRAFT_FP" id="147">
    #   <field name="east"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
    #   <field name="north"    type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
    #   <field name="up"       type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
    #   <field name="veast"    type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
    #   <field name="vnorth"   type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
    #   <field name="vup"      type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
    #   <field name="phi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
    #   <field name="theta"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
    #   <field name="psi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
    #   <field name="carrot_east"   type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
    #   <field name="carrot_north"  type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
    #   <field name="carrot_up"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
    #   <field name="carrot_psi"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
    #   <field name="thrust"        type="int32"/>
    #   <field name="flight_time"   type="uint16" unit="s"/>
    # </message>

    # <message name="INS" id="198">
    #   <field name="ins_x"     type="int32" alt_unit="m"    alt_unit_coef="0.0039063"/>
    #   <field name="ins_y"     type="int32" alt_unit="m"    alt_unit_coef="0.0039063"/>
    #   <field name="ins_z"     type="int32" alt_unit="m"    alt_unit_coef="0.0039063"/>
    #   <field name="ins_xd"    type="int32" alt_unit="m/s"  alt_unit_coef="0.0000019"/>
    #   <field name="ins_yd"    type="int32" alt_unit="m/s"  alt_unit_coef="0.0000019"/>
    #   <field name="ins_zd"    type="int32" alt_unit="m/s"  alt_unit_coef="0.0000019"/>
    #   <field name="ins_xdd"   type="int32" alt_unit="m/s2" alt_unit_coef="0.0009766"/>
    #   <field name="ins_ydd"   type="int32" alt_unit="m/s2" alt_unit_coef="0.0009766"/>
    #   <field name="ins_zdd"   type="int32" alt_unit="m/s2" alt_unit_coef="0.0009766"/>
    # </message>

    def shutdown(self):
        if self._interface is not None:
            print("Shutting down ivy interface...")
            self._interface.shutdown()
            self._interface = None

    def __del__(self):
        self.shutdown()

    def set_guided_mode(self):
        """
        change mode to GUIDED.
        """
        if self.ap_mode is not None:
            msg = PprzMessage("ground", "DL_SETTING")
            msg['ac_id'] = self.ac_id
            msg['index'] = self.ap_mode.index
            try:
                msg['value'] = self.ap_mode.ValueFromName('Guided')  # AP_MODE_GUIDED
            except ValueError:
                try:
                    msg['value'] = self.ap_mode.ValueFromName('GUIDED')  # AP_MODE_GUIDED
                except ValueError:
                    msg['value'] = 19 # fallback to fixed index
            print("Setting mode to GUIDED: %s" % msg)
            self._interface.send(msg)

    def set_nav_mode(self):
        """
        change mode to NAV.
        """
        if self.ap_mode is not None:
            msg = PprzMessage("ground", "DL_SETTING")
            msg['ac_id'] = self.ac_id
            msg['index'] = self.ap_mode.index
            try:
                msg['value'] = self.ap_mode.ValueFromName('Nav')  # AP_MODE_NAV
            except ValueError:
                try:
                    msg['value'] = self.ap_mode.ValueFromName('NAV')  # AP_MODE_NAV
                except ValueError:
                    msg['value'] = 13 # fallback to fixed index
            print("Setting mode to NAV: %s" % msg)
            self._interface.send(msg)

    def goto_ned(self, north, east, down, heading=0.0):
        """
        goto a local NorthEastDown position in meters (if already in GUIDED mode)
        """
        msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
        msg['ac_id'] = self.ac_id
        msg['flags'] = 0x00
        msg['x'] = north
        msg['y'] = east
        msg['z'] = down
        msg['yaw'] = heading
        print("goto NED: %s" % msg)
        # embed the message in RAW_DATALINK so that the server can log it
        self._interface.send_raw_datalink(msg)

    def goto_ned_relative(self, north, east, down, yaw=0.0):
        """
        goto a local NorthEastDown position relative to current position in meters (if already in GUIDED mode)
        """
        msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
        msg['ac_id'] = self.ac_id
        msg['flags'] = 0x0D
        msg['x'] = north
        msg['y'] = east
        msg['z'] = down
        msg['yaw'] = yaw
        print("goto NED relative: %s" % msg)
        self._interface.send_raw_datalink(msg)

    def goto_body_relative(self, forward, right, down, yaw=0.0):
        """
        goto to a position relative to current position and heading in meters (if already in GUIDED mode)
        """
        msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
        msg['ac_id'] = self.ac_id
        msg['flags'] = 0x0E
        msg['x'] = forward
        msg['y'] = right
        msg['z'] = down
        msg['yaw'] = yaw
        print("goto body relative: %s" % msg)
        self._interface.send_raw_datalink(msg)

    def move_at_ned_vel(self, north=0.0, east=0.0, down=0.0, yaw=0.0):
        """
        move at specified velocity in meters/sec with absolute heading (if already in GUIDED mode)
        """
        msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
        msg['ac_id'] = self.ac_id
        msg['flags'] = 0xE0
        msg['x'] = north
        msg['y'] = east
        msg['z'] = down
        msg['yaw'] = yaw
        print("move at vel NED: %s" % msg)
        self._interface.send_raw_datalink(msg)

    def move_at_body_vel(self, forward=0.0, right=0.0, down=0.0, yaw=0.0):
        """
        move at specified velocity in meters/sec with absolute heading (if already in GUIDED mode)
        """
        msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
        msg['ac_id'] = self.ac_id
        msg['flags'] = 0x62
        msg['x'] = forward
        msg['y'] = right
        msg['z'] = down
        msg['yaw'] = yaw
        print("move at vel body: %s" % msg)
        self._interface.send_raw_datalink(msg)

    # <message name="GUIDED_SETPOINT_NED" id="40" link="forwarded">
    #   <description>
    #     Set vehicle position or velocity in NED.
    #     Frame can be specified with the bits 0-3
    #     Velocity of position setpoint can be specified with the bits 5-7
    #     Flags field definition:
    #     - bit 0: x,y as offset coordinates
    #     - bit 1: x,y in body coordinates
    #     - bit 2: z as offset coordinates
    #     - bit 3: yaw as offset coordinates
    #     - bit 4: free
    #     - bit 5: x,y as vel
    #     - bit 6: z as vel
    #     - bit 7: yaw as rate
    #   </description>
    #   <field name="ac_id" type="uint8"/>
    #   <field name="flags" type="uint8">bits 0-3: frame, bits 5-7: use as velocity</field>
    #   <field name="x" type="float" unit="m">X position/velocity in NED</field>
    #   <field name="y" type="float" unit="m">Y position/velocity in NED</field>
    #   <field name="z" type="float" unit="m">Z position/velocity in NED (negative altitude)</field>
    #   <field name="yaw" type="float" unit="rad" alt_unit="deg">yaw/rate setpoint</field>
    # </message>

# class IvyRequester(object):
#     def __init__(self, interface=None):
#         self._interface = interface
#         if interface is None:
#             self._interface = IvyMessagesInterface("ivy requester")
#         self.ac_list = []

#     def __del__(self):
#         self.shutdown()

#     def shutdown(self):
#         if self._interface is not None:
#             print("Shutting down ivy interface...")
#             self._interface.shutdown()
#             self._interface = None

#     def get_aircrafts(self):

#         def aircrafts_cb(ac_id, msg):
#             self.ac_list = [int(a) for a in msg['ac_list'].split(',') if a]
#             print("aircrafts: {}".format(self.ac_list))

#         self._interface.subscribe(aircrafts_cb, "(.*AIRCRAFTS .*)")
#         sender = 'get_aircrafts'
#         request_id = '42_1' # fake request id, should be PID_index
#         self._interface.send("{} {} AIRCRAFTS_REQ".format(sender, request_id))
#         # hack: sleep briefly to wait for answer
#         sleep(0.1)
#         return self.ac_list


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

    ### End deep guidance initialization stuff
    
    
    # ac_id = args.follower_id # just for mow...

    # if args.ac_id > 0:
    #     ac_id = args.ac_id
    # else:
    #     print("No aircraft ID specified, checking available aircrafts...")
    #     interface = IvyMessagesInterface("guided mode example")
    #     req = IvyRequester(interface)
    #     # hack: sleep briefly so that connections can be established
    #     sleep(0.1)
    #     aircrafts = req.get_aircrafts()
    #     if not aircrafts:
    #         print("No active aircrafts found, aborting...")
    #         sys.exit(1)
    #     elif len(aircrafts) == 1:
    #         ac_id = aircrafts[0]
    #     else:
    #         print("multiple aircrafts found: {}".format(aircrafts))
    #         print("please specify one on the commandline...")
    #         sys.exit(1)

        try:
            g = Guidance(interface=interface, target_id=target_id, follower_id=follower_id)
            sleep(0.1)
            g.set_guided_mode()
            sleep(0.2)
            last_target_yaw = 0.0
            last_chaser_yaw = 0.0
            while True:
                # TODO: make better frequency managing
                sleep(g.step)
                # print('G IDS : ',g.ids) # debug....
                policy_input = np.zeros(8) # initializing policy input
                for rc in g.rotorcrafts:
                    rc.timeout = rc.timeout + g.step
                    # print('rc.id',rc.id)
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
                # deep guidance = [chaser_x_velocity [north], chaser_y_velocity [west], chaser_z_velocity [up], chaser_angular_velocity [counter-clockwise looking down from above]]
        
                # Send velocity command to aircraft!
                g.move_at_ned_vel(north = deep_guidance[0], east = -deep_guidance[1], down = -deep_guidance[2], yaw=-deep_guidance[3])
                print("Policy input: ", policy_input, "Deep guidance command: ", deep_guidance)
    





        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            g.set_nav_mode()
            g.shutdown()
            sleep(0.2)
            exit()


if __name__ == '__main__':
    main()



# g.goto_ned(north=2.0, east=2.0, down=-3.0, heading=radians(90))
# sleep(10)
# g.goto_ned_relative(north=-2.0, east=-2.0, down=1.0, yaw=-radians(45))
# sleep(10)
# g.goto_body_relative(forward=0.0, right=1.0, down=0.0)
# sleep(10)
# g.move_at_ned_vel(north=0.5)
# sleep(3)
# g.move_at_body_vel(forward=-0.5)
# sleep(3)
# g.set_nav_mode()
# sleep(0.2)