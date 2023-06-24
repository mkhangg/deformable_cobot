#!/usr/bin/env python
#Enable robot before running test
# rosrun baxter_tools enable_robot.py -e
import rospy
import baxter_interface
import threading
import math
import numpy as np
import socket
import json
import signal
import time
import os
import struct
import rospy
from std_srvs.srv import Empty
from reflex_msgs.msg import PoseCommand
from geometry_msgs.msg import Twist
from commands import *
import tf
import sys

def timer_callback(event):
    global twist_output
    pub.publish(twist_output)

def base_move(twist, duration):
    global twist_output, zero_twist
    #rospy.loginfo("base_move", twist, duration)
    twist_output = twist
    rospy.sleep(duration)
    #twist_output = zero_twist
    #rospy.sleep(1.0)

def deg2rad(deg):
    return ((deg*math.pi)/180.0)
    
def rad2deg(rad):
    return ((rad/math.pi)*180.0)

def handler(signum, frame):
    rospy.loginfo("STOP signal was received! (Ctr+C)")
    global b_running
    b_running = False

def move_thread(limb, name, joint_pos):
    rospy.loginfo(name, ' move to ', joint_pos)
    limb.move_to_joint_positions(joint_pos)
    rospy.loginfo(name, " .Done!")

#Client has commands: GetPosLeft:NoData,GetPosRight:NoData, 
# SetPosL:DictionaryData, SetPosR:DictionaryData

def zero_hands_thread(pos_pub_left, pos_pub_right):
    pos_pub_left.publish(PoseCommand(0, 0, 0, 0)),
    pos_pub_right.publish(PoseCommand(0, 0, 0, 0))

def cmd_thread(cmd):
    cmd_str = ""
    if cmd == ENBABLE_ROBOT:
        cmd_str = "rosrun baxter_tools enable_robot.py -e"
    if cmd == DISABLE_ROBOT:
        cmd_str = "rosrun baxter_tools enable_robot.py -d"
    if cmd == TUCK_ROBOT:
        cmd_str = "rosrun baxter_tools tuck_arms.py -t"
    if cmd == UNTUCK_ROBOT:
        cmd_str = "rosrun baxter_tools tuck_arms.py -u"
    rospy.loginfo("Thread CMD = %s", cmd_str)
    os.system(cmd_str)

rospy.init_node('Baxter_Move_Client')
calibrate_fingers_hand1 = rospy.ServiceProxy('/reflex_takktile/calibrate_fingers', Empty)
calibrate_tactile_hand1 = rospy.ServiceProxy('/reflex_takktile/calibrate_tactile', Empty)
calibrate_fingers_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/calibrate_fingers', Empty)
calibrate_tactile_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/calibrate_tactile', Empty)
pos_pub_right = rospy.Publisher('/reflex_takktile/command_position', PoseCommand, queue_size=1)
pos_pub_left = rospy.Publisher('/hand2/reflex_takktile/command_position', PoseCommand, queue_size=1)

#Hand Pose
close_angle = 125
preshape_angle = 0  #old value: 60
p_open =  PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(preshape_angle))
p_close = PoseCommand(deg2rad(close_angle), deg2rad(close_angle), deg2rad(close_angle), deg2rad(preshape_angle))

speed = 0.15
dist = 0.15
vel = Twist()

source_link = '/world'
listener = tf.TransformListener()

left_limb = baxter_interface.limb.Limb('left')
right_limb = baxter_interface.limb.Limb('right')
lj = left_limb.joint_names()
rj = right_limb.joint_names()

def exec_cmd(cmd):    
    print('exec_cmd = ', cmd)
    # rosrun tf tf_echo /world /left_hand
    if cmd == GET_WLPOS:
        try:
            #base_link
            (trans,rot) = listener.lookupTransform(source_link, '/left_hand', rospy.Time(0))
        except Exception as e:
            trans = [0, 0, 0]
            rot = [0, 0, 0, 0]
            print(e)
        euler = tf.transformations.euler_from_quaternion(rot)
        data = trans + rot + list(euler) #Merge two lists
        print('GET_LRPOS = ', data)
  
    elif cmd == GET_WRPOS:
        try:
            (trans,rot) = listener.lookupTransform(source_link, '/right_hand', rospy.Time(0))
        except Exception as e:
            trans = [0, 0, 0]
            rot = [0, 0, 0, 0]
            print(e)
        euler = tf.transformations.euler_from_quaternion(rot)
        data = trans + rot + list(euler) #Merge two lists
        print('GET_WRPOS = ', data)
    elif cmd == ENBABLE_ROBOT or cmd == DISABLE_ROBOT or cmd == TUCK_ROBOT or cmd == UNTUCK_ROBOT:
        client_thread = threading.Thread(
            target=cmd_thread,
            args=(cmd,)
        )
        client_thread.start()
        
        
    elif cmd == CALIBRATE_HANDS:
        rospy.loginfo("CALIBRATE_HANDS")
        calibrate_fingers_hand1()
        calibrate_tactile_hand1()
        calibrate_fingers_hand2()
        calibrate_tactile_hand2()
        pass

    elif cmd == OPEN_LEFT_HAND:
        pos_pub_left.publish(p_open)
        rospy.loginfo("OPEN_LEFT_HAND")
        pass
        
    elif cmd == CLOSE_LEFT_HAND:
        pos_pub_left.publish(p_close)
        rospy.loginfo("CLOSE_LEFT_HAND")
        pass
        
    elif cmd == OPEN_RIGHT_HAND:
        pos_pub_right.publish(p_open)
        rospy.loginfo("OPEN_RIGHT_HAND")
        pass
        
    elif cmd == CLOSE_RIGHT_HAND:
        pos_pub_right.publish(p_close)
        rospy.loginfo("CLOSE_RIGHT_HAND")
        pass

    elif cmd == ZERO_HANDS:
        #pos_pub_left.publish(PoseCommand(0, 0, 0, 0))
        #pos_pub_right.publish(PoseCommand(0, 0, 0, 0))
        
        zero_hands_thread = threading.Thread(
            target=lambda :( 
               pos_pub_left.publish(PoseCommand(0, 0, 0, 0)),
               pos_pub_right.publish(PoseCommand(0, 0, 0, 0))
            ),
            args=()
        )
        zero_hands_thread.start()
        rospy.loginfo("ZERO_HANDS")
        pass
        
    #Jogging
    elif LEFT_S0_NAG <= cmd and cmd <= RIGHT_W2_POS:
       
        def set_j(limb, joint_name, delta):
            current_position = limb.joint_angle(joint_name)
            joint_command = {joint_name: current_position + delta}
            print('joint_command = ', joint_command, ' from current = ', current_position)
            limb.move_to_joint_positions(joint_command)
            # rospy.sleep(1)
            
        bindings = {
            #key: (function, args, description)
            21: (set_j, [left_limb, lj[0], 0.1], "left_s0 increase"),
            22: (set_j, [left_limb, lj[0], -0.1], "left_s0 decrease"),
            23: (set_j, [left_limb, lj[1], 0.1], "left_s1 increase"),
            24: (set_j, [left_limb, lj[1], -0.1], "left_s1 decrease"),
            25: (set_j, [left_limb, lj[2], 0.1], "left_e0 increase"),
            26: (set_j, [left_limb, lj[2], -0.1], "left_e0 decrease"),
            27: (set_j, [left_limb, lj[3], 0.1], "left_e1 increase"),
            28: (set_j, [left_limb, lj[3], -0.1], "left_e1 decrease"),
            29: (set_j, [left_limb, lj[4], 0.1], "left_w0 increase"),
            30: (set_j, [left_limb, lj[4], -0.1], "left_w0 decrease"),
            31: (set_j, [left_limb, lj[5], 0.1], "left_w1 increase"),
            32: (set_j, [left_limb, lj[5], -0.1], "left_w1 decrease"),
            33: (set_j, [left_limb, lj[6], 0.6], "left_w2 increase"),
            34: (set_j, [left_limb, lj[6], -0.6], "left_w2 decrease"),
            # ',': (grip_left.close, [], "left: gripper close"),
            # 'm': (grip_left.open, [], "left: gripper open"),
            # '/': (grip_left.calibrate, [], "left: gripper calibrate"),

            41: (set_j, [right_limb, rj[0], 0.1], "right_s0 increase"),
            42: (set_j, [right_limb, rj[0], -0.1], "right_s0 decrease"),
            43: (set_j, [right_limb, rj[1], 0.1], "right_s1 increase"),
            44: (set_j, [right_limb, rj[1], -0.1], "right_s1 decrease"),
            45: (set_j, [right_limb, rj[2], 0.1], "right_e0 increase"),
            46: (set_j, [right_limb, rj[2], -0.1], "right_e0 decrease"),
            47: (set_j, [right_limb, rj[3], 0.1], "right_e1 increase"),
            48: (set_j, [right_limb, rj[3], -0.1], "right_e1 decrease"),
            49: (set_j, [right_limb, rj[4], 0.1], "right_w0 increase"),
            50: (set_j, [right_limb, rj[4], -0.1], "right_w0 decrease"),
            51: (set_j, [right_limb, rj[5], 0.1], "right_w1 increase"),
            52: (set_j, [right_limb, rj[5], -0.1], "right_w1 decrease"),
            53: (set_j, [right_limb, rj[6], 0.1], "right_w2 increase"),
            54: (set_j, [right_limb, rj[6], -0.1], "right_w2 decrease"),
            # 'c': (grip_right.close, [], "right: gripper close"),
            # 'x': (grip_right.open, [], "right: gripper open"),
            # 'b': (grip_right.calibrate, [], "right: gripper calibrate"),
        }
        
        jcmd = bindings[cmd]
        jcmd[0](*jcmd[1])
        # rospy.loginfo("command: %s" % (jcmd[2],))

    elif cmd == BASE_MOVE_FORWARD:
        rospy.loginfo("BASE_MOVE_FORWARD")
        vel.linear.x = speed
        vel.linear.y = 0
        vel.angular.z = 0
        base_move(vel, dist/speed)
       

    elif cmd == BASE_MOVE_BACKWARD:
        rospy.loginfo("BASE_MOVE_BACKWARD")
        vel.linear.x = -speed
        vel.linear.y = 0
        vel.angular.z = 0
        base_move(vel, dist/speed)
        
        
    elif cmd == BASE_MOVE_LEFT:
        rospy.loginfo("BASE_MOVE_LEFT")
        vel.linear.x = 0
        vel.linear.y = speed
        vel.angular.z = 0
        base_move(vel, dist/speed)
        
        
    elif cmd == BASE_MOVE_RIGHT:
        rospy.loginfo("BASE_MOVE_RIGHT")
        vel.linear.x = 0
        vel.linear.y = -speed
        vel.angular.z = 0
        base_move(vel, dist/speed)
        
    elif cmd == BASE_TURN_LEFT:
        rospy.loginfo("BASE_TURN_LEFT")
        vel.linear.x = 0
        vel.linear.y = 0
        vel.angular.z = speed
        base_move(vel, dist/speed)
        
    elif cmd == BASE_TURN_RIGHT:
        rospy.loginfo("BASE_TURN_RIGHT")
        vel.linear.x = 0
        vel.linear.y = 0
        vel.angular.z = -speed
        base_move(vel, dist/speed)

    elif cmd == BASE_STOP:
        rospy.loginfo("BASE_STOP")
        vel.linear.x = 0
        vel.linear.y = 0
        vel.angular.z = 0
        base_move(vel, dist/speed)
            

###################
b_running = True
def handler(signum, frame):
    rospy.loginfo("STOP signal was received! (Ctr+C)")
    global b_running
    b_running = False
    

import socket
import signal

cmd = int(sys.argv[1])

signal.signal(signal.SIGINT, handler)



if cmd < 100:
    exec_cmd(int(sys.argv[1]))
else:
    HOST = "192.168.1.249"  # The server's hostname or IP address
    PORT = int(sys.argv[2])
    t = 2

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)	
    print('Waiting for a connection')

    # while b_running:
    conn, addr = s.accept()
    print("Connected by ", addr)
    if cmd == 100: #rotate right_s0+
        for i in range(3):
            exec_cmd(42)
            rospy.sleep(1)
    elif cmd == 101:    #left_w2-
        for i in range(-1, 4):
            print('i = ', i)
            if i < 2:
                conn.send('1')
                rospy.sleep(t)
            else:
                exec_cmd(33)
                rospy.sleep(t)
                conn.send('1')
                rospy.sleep(t)
    elif cmd == 102:    #left_w2+
        for i in range(-1, 4):
            print('i = ', i)
            if i < 2:
                rospy.sleep(t)
                conn.send('1')
            else:
                exec_cmd(34)
                rospy.sleep(t)
                conn.send('1')
                rospy.sleep(t)
    elif cmd == 103:    #Test communication
        for i in range(2):
            conn.send('1')
            rospy.sleep(t)
            if i == 0:
                pass
            else:
                pass
