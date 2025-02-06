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
    
    #Taking the values for tau and delta_t from GLISp algorithm, 
    # they are parameters to optimize in open loop controller case
    #input_tau = input("Please enter value for tau: ")
    #tau= float(input_tau)
    #input_dt = input("Please enter value for delta_t: ")
    #dt= float(input_dt)
    #req.tau = tau
    #req.dt = dt
    req.tau = 0.1
    req.dt = 0.05
    #We can also add parameters ay and by to GLISp algorithm
    #input_ay = input("Please enter value for ay: ")
    #ay= float(input_ay)
    #input_by = input("Please enter value for by: ")
    #by= float(input_by)
    #req.ay = ay
    #req.by = by
    req.ay = 25.0
    req.by = 6.25

    """
    # Compose request message - Marko -ramp trajectory
    req.dmp_name = "/home/ros/projects/merlin_ws/src/dmps_thesis/example.yaml"
    #req.tau = 0.1
    #req.dt = 0.1
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
    """

    #Trajectory generated from teleoperation - Christian 
    # Compose request message
    req.dmp_name = "/home/ros/projects/merlin_ws/src/dmps_thesis/MyWeights.yaml"
    #req.tau = rospy.get_param("/tau")
    #req.dt = rospy.get_param("/dt")
    #rospy.loginfo(f"Using tau: {req.tau} and dt: {req.dt}")

    # Read the file for the trajectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    file_path = os.path.join(script_dir, "../reference_trajectory.csv")
    #file_path = os.path.join(script_dir, "../reference_trajectory_old.csv")
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

    rospy.loginfo(f"The inital pose is: {req.initial_pose}")
    rospy.loginfo(f"The goal pose is: {req.goal_pose}")

    try:
        service_client = rospy.ServiceProxy('/generate_motion_service', GenerateMotion)
        rospy.loginfo(service_client(req))
    except :
        rospy.loginfo("Service call failed")
    
