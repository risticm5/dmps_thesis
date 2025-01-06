#!/usr/bin/env python3
import numpy as np 
import rospy
from ros_dmp import RollDmp
from ros_dmp.srv import *
from ros_dmp.msg import *
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose
import tf

# This is node I made to generate trajectory with learned DMPs
# In this case, I want to have constant online(synchronous) communication, not Service communicatio that already exists 
# in the script generate_motion_service_node.py

def generate_motion(initial_pose, goal_pose, file_name, tau, dt):
    rospy.loginfo("Received motion generation request")
    # Initial pose
    rpy = tf.transformations.euler_from_quaternion([initial_pose.pose.orientation.x, initial_pose.pose.orientation.y,
                                                                    initial_pose.pose.orientation.z, initial_pose.pose.orientation.w])
    initial_pose = np.array([initial_pose.pose.position.x, initial_pose.pose.position.y, initial_pose.pose.position.z,
                            rpy[0], rpy[1], rpy[2]])
        
    # Goal Pose
    rpy = tf.transformations.euler_from_quaternion([goal_pose.pose.orientation.x, goal_pose.pose.orientation.y,
                                                                    goal_pose.pose.orientation.z, goal_pose.pose.orientation.w])
    goal_pose = np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z,
                            rpy[0], rpy[1], rpy[2]])

    rospy.loginfo("Generating motion for dmp " + file_name)
    dmp = RollDmp(file_name, dt) #initialization of learned DMPs 
    pos, vel, acc = dmp.roll(goal_pose, initial_pose, tau) #execution, trajectory generation, of learned DMPs

    # Publish cartesian trajectory
    cartesian_trajectory = CartesianTrajectory()
    cartesian_trajectory.header.frame_id = "base_link"
    path = Path()
    path.header.frame_id = "base_link"
    for i in range(pos.shape[0]):
        x, y, z, w = tf.transformations.quaternion_from_euler(pos[i, 3], pos[i, 4], pos[i, 5])
        pose = Pose()
        pose.position.x = pos[i, 0]
        pose.position.y = pos[i, 1]
        pose.position.z = pos[i, 2]
        pose.orientation.x = x
        pose.orientation.y = y
        pose.orientation.z = z
        pose.orientation.w = w
            
        cartesian_state = CartesianState()
        cartesian_state.pose = pose
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose

        cartesian_state.vel.linear.x = vel[i, 0]
        cartesian_state.vel.linear.y = vel[i, 1]
        cartesian_state.vel.linear.z = vel[i, 2]
        cartesian_state.vel.angular.x = vel[i, 3]
        cartesian_state.vel.angular.y = vel[i, 4]
        cartesian_state.vel.angular.z = vel[i, 5]

        cartesian_state.acc.linear.x = acc[i, 0]
        cartesian_state.acc.linear.y = acc[i, 1]
        cartesian_state.acc.linear.z = acc[i, 2]
        cartesian_state.acc.angular.x = acc[i, 3]
        cartesian_state.acc.angular.y = acc[i, 4]
        cartesian_state.acc.angular.z = acc[i, 5]

        cartesian_trajectory.cartesian_state.append(cartesian_state)
        path.poses.append(pose_stamped)
        
        trajectory_pub.publish(cartesian_trajectory)
        path_pub.publish(path)
    rospy.loginfo("Motion generated and published on respective topics")


if __name__ == "__main__":
    
    rospy.init_node("generate_motion_node_marko")
    rospy.loginfo("Started Motion Generation")
    
    # Publishers
    trajectory_pub = rospy.Publisher('~cartesian_trajectory_marko', CartesianTrajectory, queue_size=1)
    path_pub = rospy.Publisher('~cartesian_path_marko', Path, queue_size=1)
    
    #Motion generation parameters
    file_name = "/home/ros/projects/merlin_ws/src/dmps_thesis/example.yaml"
    tau = 1.0
    dt = 0.01
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = "base_link"
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "base_link"
    goal_pose.pose.position.x = 1.0
    goal_pose.pose.position.y = 0.0
    goal_pose.pose.position.z = 1.0
    goal_pose.pose.orientation.x = 1.0
    goal_pose.pose.orientation.y = 1.0
    goal_pose.pose.orientation.z = 1.0
    goal_pose.pose.orientation.w = 1.0
    
    generate_motion(initial_pose, goal_pose, file_name, tau, dt)
    rospy.spin()
    #rospy.loginfo("Ended Motion Generation")