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
    req.dmp_name = "dmp/weights/example.yaml"
    req.dmp_name = "example.yaml"
    req.tau = 2.0
    req.dt = 0.01
    req.goal_pose = PoseStamped()
    req.goal_pose.header.frame_id = "base_link"
    req.goal_pose.pose.position.x = 0.3 #0.8
    req.goal_pose.pose.position.y = 0.0
    req.goal_pose.pose.position.z = 0.3  #0.3*np.sin(20*0.8)
    x1, y1, z1, w1 = tf.transformations.quaternion_from_euler(0.0, 0.0, 3.14/4)
    req.goal_pose.pose.orientation.x = x1 #0.0
    req.goal_pose.pose.orientation.y = y1 #0.0
    req.goal_pose.pose.orientation.z = z1 #0.0
    req.goal_pose.pose.orientation.w = w1 #1.0 #for zero angle
    req.initial_pose.header.frame_id = "base_link"

    try:
        service_client = rospy.ServiceProxy('/generate_motion_service', GenerateMotion)
        rospy.loginfo(service_client(req))
    except :
        rospy.loginfo("Service call failed")
    
