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

    # Compose service request
    req.header.frame_id = 'dmp_ref'
    req.output_weight_file_name = 'MyWeights.yaml'
    req.dmp_name = 'reference_trajectory_line'
    req.header.stamp = rospy.Time.now()
    req.n_bfs = rospy.get_param("/n_radial_basis")
    req.n_dmps = rospy.get_param("/n_dmps")

    # Read trajectory data from CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    file_path = os.path.join(script_dir, "../reference_trajectory_line.csv")
    
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
