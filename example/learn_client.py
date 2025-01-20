#!/usr/bin/env python3
import numpy as np 
import rospy
from geometry_msgs.msg import Pose
from ros_dmp.srv import *
import tf

if __name__ == "__main__":
    ''' 
    Generate the fake reference trajectory and call the learn_dmp service.
    As of now, the reference frame is dmp_ref and it is alligned as the frame 'world' of the ur5e robot.
    This file is going to be changed a bit: the reference trajectory is going to be read from a file.
    '''

    rospy.init_node('learn_dmp_service_test_client')
    req = LearnDMPRequest()

    # Compose service request
    req.header.frame_id = 'dmp_ref'
    req.output_weight_file_name = 'example.yaml'
    req.dmp_name = 'reference_trajectory'
    req.header.stamp = rospy.Time.now()
    req.n_bfs = rospy.get_param("/n_radial_basis")
    req.n_dmps = rospy.get_param("/n_dmps")

    # Case 1: line in space
    x_start, y_start, z_start, w_start = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    x_end, y_end, z_end, w_end = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    x = np.linspace(0, -0.2, 100)
    y = np.linspace(0, 0.2, 100)
    z = np.linspace(0, 0.1, 100)
    o_x = np.linspace(x_start, x_end, 100)
    o_y = np.linspace(y_start, y_end, 100)
    o_z = np.linspace(z_start, z_end, 100)
    o_w = np.linspace(w_start, w_end, 100)
    

    # Case 2: square wave
    '''
    x_start, y_start, z_start, w_start = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    x_end, y_end, z_end, w_end = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    x = np.linspace(0, 0.2, 100)
    y = np.linspace(0, 0.2, 100)
    z = np.zeros(20)
    z = np.hstack((z, np.ones(60) * 0.2))
    z = np.hstack((z, np.ones(20) * 0.1))
    o_x = np.linspace(x_start, x_end, 100)
    o_y = np.linspace(y_start, y_end, 100)
    o_z = np.linspace(z_start, z_end, 100)
    o_w = np.linspace(w_start, w_end, 100)
    '''

    # Generate the full pose
    for i in range(x.shape[0]):

        # All these values of the pose are specified with respect to dmp_ref frame
        pose = Pose()
        pose.position.x = x[i]
        pose.position.y = y[i]
        pose.position.z = z[i]
        pose.orientation.x = o_x[i]
        pose.orientation.y = o_y[i]
        pose.orientation.z = o_z[i]
        pose.orientation.w = o_w[i]
        req.poses.append(pose)

    # Call the service
    try:
        service_client = rospy.ServiceProxy('/learn_dynamic_motion_primitive_service', LearnDMP)
        rospy.loginfo(service_client(req))
    except :
        rospy.loginfo("Service call failed")
