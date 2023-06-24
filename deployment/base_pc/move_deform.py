#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
#import geometry_msgs.msg.Quaternion
from reflex_msgs.msg import PoseCommand
import tf
import copy
from std_msgs.msg import String

import rospy
from std_msgs.msg import Float32MultiArray
from tf import TransformListener
import tf
import signal
import math
import threading

#Global variables
b_running = True
uvd = [-1, -1, -1]
b_read = False

def cb(data):
    global uvd, b_read
    uvd = data.data
    b_read = True
    #rospy.loginfo(rospy.get_caller_id(), " : ")
    #rospy.loginfo(uvd)  


def handler(signum, frame):
    rospy.loginfo("STOP signal was received! (Ctr+C)")
    global b_running
    b_running = False


def deg2rad(deg):
    return ((deg*math.pi)/180.0)

b_move_end = False

#(x = 0.77, y=-0.07, z=1.07, bound = 0.22)
#(x = 0.72, y=-0.07, z=1.08, bound = 0.20)
#(x = 0.67, y=-0.07, z=1.10, bound = 0.18)
#(x = 0.62, y=-0.07, z=1.12, bound = 0.16)
#(x = 0.57, y=-0.05, z=1.15, bound = 0.12)
#(x = 0.52, y=-0.05, z=1.17, bound = 0.10)
delta = 0.02 
bound = 0.12
base_x = 0.67 
base_y = -0.07
base_z = 1.15

points = []
range_y, range_z = bound, bound
m, n = int(range_y/delta), int(range_z/delta)
print('m = ', m, ', n = ', n, ', mxn = ', m*n)
for i in range(m): #dy
    for j in range(n): #dz
        #Change to root yz plane coordinate
        y = base_y
        z = base_z
        p = [base_x, y+delta*i, z + delta*j]
        points.append(p)
        
start = 0
stop = 1
# stop = m*n
f = open('calib_d' + str(base_x) +  '.csv', 'w')
f.write('u, v, d, x, y, z \n') #Write header

def move_arm(p):
    global b_move_end
    b_move_end = False
    print('move_arm = ', p)
    
    arm_name = "left_arm"
    end_effector = "left_hand"

    #print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    #rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    arm = moveit_commander.MoveGroupCommander(arm_name)
    #arm.set_planning_frame('base_link')	
    #print("============ Reference frame: %s" % arm.get_planning_frame())
    arm.set_end_effector_link(end_effector)
    #print("============ End effector frame: %s" % arm.get_end_effector_link())

    robot = moveit_commander.RobotCommander()
    #print("============ Robot Groups: ", robot.get_group_names())
    
    pose_target = geometry_msgs.msg.Pose()
    pose_target.position.x = p[0]
    pose_target.position.y = p[1]  
    pose_target.position.z = p[2]
    
    q = tf.transformations.quaternion_from_euler(deg2rad(0), deg2rad(0), deg2rad(0))   
    pose_target.orientation.x = q[0]
    pose_target.orientation.y = q[1]
    pose_target.orientation.z = q[2]
    pose_target.orientation.w = q[3]
    
    print(pose_target)
    arm.set_pose_target(pose_target, end_effector_link = end_effector)
    traj = arm.plan()
    #print("len traj 1 = ", len(traj.joint_trajectory.points))
    arm.execute(traj)
    rospy.sleep(1)
    b_move_end = True
    #rospy.sleep(5)


#====MAIN=================
rospy.init_node('listener')
listener = TransformListener()
rate = rospy.Rate(1) #HZ
signal.signal(signal.SIGINT, handler)
rospy.Subscriber("uvd", Float32MultiArray, cb)
trans = [-1, -1, -1]
source_link = '/world'
dest_link = '/left_hand'



# while b_running:
print('start = ', start , ' = ', points[start])
move_arm(points[start])
# rospy.spin()
    # while b_running:
        # try:
           
        # except tf.Exception as e :
           
    # start += 1
    # if start >= stop:
        # break
moveit_commander.roscpp_shutdown()
print "STOPPING...."
print('Exit....')