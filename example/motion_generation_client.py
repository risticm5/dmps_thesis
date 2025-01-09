#!/usr/bin/env python3
import numpy as np 
import rospy
from geometry_msgs.msg import PoseStamped
from ros_dmp.srv import *
import tf

if __name__ == "__main__":

    rospy.init_node('generate_motion_service_test_client')
    req = GenerateMotionRequest()

    # Compose request message
    req.dmp_name = "/home/christian/projects/merlin_ws/src/dmps_thesis/example.yaml"
    #req.dmp_name = "example.yaml"
    req.tau = 0.1
    req.dt = 0.01
    
    # Define the initial pose
    req.initial_pose = PoseStamped()
    req.initial_pose.header.frame_id = "dmp_ref"
    req.initial_pose.pose.position.x = 0.0 #0.8
    req.initial_pose.pose.position.y = 0.0
    req.initial_pose.pose.position.z = 0.0  #0.3*np.sin(20*0.8)
    x1, y1, z1, w1 = tf.transformations.quaternion_from_euler(1.57, 0.0, 0.0) # rx, ry, rz
    req.initial_pose.pose.orientation.x = x1 
    req.initial_pose.pose.orientation.y = y1 
    req.initial_pose.pose.orientation.z = z1 
    req.initial_pose.pose.orientation.w = w1 

    # Define the goal pose
    req.goal_pose = PoseStamped()
    req.goal_pose.header.frame_id = "dmp_ref"
    req.goal_pose.pose.position.x = 0.2 #0.8
    req.goal_pose.pose.position.y = 0.2
    req.goal_pose.pose.position.z = 0.1  #0.3*np.sin(20*0.8)
    x2, y2, z2, w2 = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    req.goal_pose.pose.orientation.x = x2 
    req.goal_pose.pose.orientation.y = y2 
    req.goal_pose.pose.orientation.z = z2 
    req.goal_pose.pose.orientation.w = w2 

    
    try:
        service_client = rospy.ServiceProxy('/generate_motion_service', GenerateMotion)
        rospy.loginfo(service_client(req))
    except :
        rospy.loginfo("Service call failed")
    
