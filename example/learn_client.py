#!/usr/bin/env python3
import numpy as np 
import rospy
from geometry_msgs.msg import Pose
from ros_dmp.srv import *
import tf

if __name__ == "__main__":
    ''' 
    Generate the fake reference trajectory and call the learn_dmp service
    '''

    rospy.init_node('learn_dmp_service_test_client')
    req = LearnDMPRequest()

    # Compose service request
    req.header.frame_id = 'dmp_ref'
    req.output_weight_file_name = 'example.yaml'
    req.dmp_name = 'reference_trajectory'
    req.header.stamp = rospy.Time.now()
    req.n_bfs = 500 #500
    req.n_dmps = 6

    # Generating a 'fake' reference trajectory
    x = np.linspace(0, 0.2, 100)
    y = np.linspace(0, 0.2, 100)
    #z = np.zeros(20)
    #z = np.hstack((z, np.ones(60) * 0.2))
    #z = np.hstack((z, np.ones(20) * 0.1))
    z = np.linspace(0, 0.1, 100)
    x_start, y_start, z_start, w_start = tf.transformations.quaternion_from_euler(1.57, 0.0, 0.0) # rx, ry, rz
    x_end, y_end, z_end, w_end = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
    o_x = np.linspace(x_start, x_end, 100) #np.zeros(50)
    o_y = np.linspace(y_start, y_end, 100) #np.zeros(50)
    o_z = np.linspace(z_start, z_end, 100) #np.zeros(50)
    o_w = np.linspace(w_start, w_end, 100) #np.ones(50) #for zero angle

    
    #default dt=0.01

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
