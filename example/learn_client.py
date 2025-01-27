#!/usr/bin/env python3
import numpy as np 
import rospy
from geometry_msgs.msg import Pose
from ros_dmp.srv import *
import tf
import csv
import sys 
import os
import math

def normalize_quaternion(x, y, z, w):
    """Normalize the quaternion to ensure its magnitude is 1."""
    magnitude = math.sqrt(x**2 + y**2 + z**2 + w**2)
    return x / magnitude, y / magnitude, z / magnitude, w / magnitude

def ensure_smaller_axis_angle(x, y, z, w):
    """
    Ensure the quaternion represents the smaller axis-angle rotation.
    If the angle is greater than π, invert the quaternion.
    """
    # Normalize the quaternion
    x, y, z, w = normalize_quaternion(x, y, z, w)
    
    # Calculate the rotation angle
    theta = 2 * math.acos(w)
    
    # If the rotation angle exceeds π, invert the quaternion
    if theta > math.pi:
        x, y, z, w = -x, -y, -z, -w
    
    return x, y, z, w

if __name__ == "__main__":

    rospy.init_node('learn_dmp_service_test_client')
    req = LearnDMPRequest()
    """
    # Generating a hypothetical trajectory - Marko - just a simple ramp (or sin wave)
    x = np.linspace(0, 0.3) #x = np.linspace(0, 0.8)
    y = np.zeros(50)
    z = np.linspace(0,0.3)  #z = 0.3*np.sin(20*x)
    #otientation
    x1, y1, z1, w1 = tf.transformations.quaternion_from_euler(0.0, 0.0, 3.14/4)
    o_x = np.linspace(0,x1) 
    o_y = np.linspace(0,y1) 
    o_z = np.linspace(0,z1) 
    o_w = np.linspace(0,w1)

    # Compose service request
    req.header.frame_id = 'base_link'
    req.output_weight_file_name = 'example.yaml'
    req.dmp_name = 'marko_traj'
    req.header.stamp = rospy.Time.now()
    req.n_bfs = 100 #500
    req.n_dmps = 6
    #default dt=0.01

    for i in range(x.shape[0]):
        pose = Pose()
        pose.position.x = x[i]
        pose.position.y = y[i]
        pose.position.z = z[i]
        pose.orientation.x = o_x[i]
        pose.orientation.y = o_y[i]
        pose.orientation.z = o_z[i]
        pose.orientation.w = o_w[i]
        req.poses.append(pose)    
    """

    #Trajectory learned from teleoperation - Christian 
    ''' 
    Generate the reference trajectory from a CSV file and call the learn_dmp service.
    The reference frame is dmp_ref, aligned with the 'world' frame of the UR5e robot.
    '''

    # Compose service request
    req.header.frame_id = 'dmp_ref'
    req.output_weight_file_name = 'MyWeights.yaml'
    req.dmp_name = 'reference_trajectory'
    req.header.stamp = rospy.Time.now()
    req.n_bfs = rospy.get_param("/n_radial_basis")
    req.n_dmps = rospy.get_param("/n_dmps")

    # Read trajectory data from CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    file_path = os.path.join(script_dir, "../reference_trajectory.csv")
    
    # Initialize the poses
    poses = []
    
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Assuming CSV columns: position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w
                pose = Pose()
                pose.position.x = float(row[1])
                pose.position.y = float(row[2])
                pose.position.z = float(row[3])

                # Extract and process quaternion values
                qx, qy, qz, qw = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                qx, qy, qz, qw = ensure_smaller_axis_angle(qx, qy, qz, qw)

                pose.orientation.x = qx
                pose.orientation.y = qy
                pose.orientation.z = qz
                pose.orientation.w = qw
                poses.append(pose)
    except FileNotFoundError:
        rospy.logerr(f"File not found.")
        exit(1)
    except ValueError as e:
        rospy.logerr(f"Error parsing row: {row}. {e}")
        exit(1)
    except Exception as e:
        rospy.logerr(f"Error reading trajectory file: {e}")
        exit(1)

    # Add poses to request
    req.poses = poses

    # Call the service
    try:
        service_client = rospy.ServiceProxy('/learn_dynamic_motion_primitive_service', LearnDMP)
        rospy.loginfo(service_client(req))
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")