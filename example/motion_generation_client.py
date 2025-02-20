#!/usr/bin/env python3
import numpy as np 
import rospy
from geometry_msgs.msg import PoseStamped
from ros_dmp.srv import *
import tf
import os
import csv

if __name__ == "__main__":

    rospy.init_node('generate_motion_service_test_client')
    req = GenerateMotionRequest()

    # Compose request message
    req.dmp_name = "/home/christian/projects/merlin_ws/src/dmps_thesis/MyWeights_c_shape.yaml"
    #req.dmp_name = "example.yaml"
    #req.tau = 0.1
    req.tau = rospy.get_param("/tau")
    req.dt = rospy.get_param("/dt")
    #req.tau = 0.1
    #req.dt = 0.02
    rospy.loginfo(f"Using tau: {req.tau} and dt: {req.dt}")

    # Read the file for the trajectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    file_path = os.path.join(script_dir, "../reference_trajectory_c_shape.csv")

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        row_init = None
        row_goal = None
        for k, row in enumerate(reader):
            if k == 0:
                row_init = row
            row_goal = row

    
    # Define the initial pose
    req.initial_pose = PoseStamped()
    req.initial_pose.header.frame_id = "dmp_ref"
    req.initial_pose.pose.position.x = float(row_init[1])
    req.initial_pose.pose.position.y = float(row_init[2])
    req.initial_pose.pose.position.z = float(row_init[3])
    #x1, y1, z1, w1 = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    req.initial_pose.pose.orientation.x = float(row_init[4])
    req.initial_pose.pose.orientation.y =float(row_init[5])
    req.initial_pose.pose.orientation.z = float(row_init[6])
    req.initial_pose.pose.orientation.w = float(row_init[7])

    # Define the goal pose
    req.goal_pose = PoseStamped()
    req.goal_pose.header.frame_id = "dmp_ref"
    req.goal_pose.pose.position.x = float(row_goal[1])
    req.goal_pose.pose.position.y = float(row_goal[2])
    req.goal_pose.pose.position.z = float(row_goal[3])
    #x2, y2, z2, w2 = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    req.goal_pose.pose.orientation.x = float(row_goal[4])
    req.goal_pose.pose.orientation.y = float(row_goal[5])
    req.goal_pose.pose.orientation.z = float(row_goal[6])
    req.goal_pose.pose.orientation.w = float(row_goal[7])
  
    try:
        service_client = rospy.ServiceProxy('/generate_motion_service', GenerateMotion)
        rospy.loginfo(service_client(req))
    except :
        rospy.loginfo("Service call failed")
    
