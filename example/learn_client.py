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

    # Generating a 'fake' reference trajectory
    x = np.linspace(0, 0.3, 100)
    y = np.linspace(0, 0.3, 100)
    #z = np.zeros(20)
    #z = np.hstack((z, np.ones(60) * 0.2))
    #z = np.hstack((z, np.ones(20) * 0.1))
    z = np.linspace(0, 0.2, 100)
    x1, y1, z1, w1 = tf.transformations.quaternion_from_euler(0.0, 0.0, 3.14)
    o_x = np.linspace(0, x1, 100) #np.zeros(50)
    o_y = np.linspace(0, y1, 100) #np.zeros(50)
    o_z = np.linspace(0, z1, 100) #np.zeros(50)
    o_w = np.linspace(0, w1, 100) #np.ones(50) #for zero angle

    # Compose service request
    req.header.frame_id = 'base_link'
    req.output_weight_file_name = 'example.yaml'
    req.dmp_name = 'reference_trajectory'
    req.header.stamp = rospy.Time.now()
    req.n_bfs = 500 #500
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

    # Call the service
    try:
        service_client = rospy.ServiceProxy('/learn_dynamic_motion_primitive_service', LearnDMP)
        rospy.loginfo(service_client(req))
    except :
        rospy.loginfo("Service call failed")
