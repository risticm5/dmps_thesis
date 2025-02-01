#!/usr/bin/env python3
import rospy
import numpy as np
from ros_dmp import RollDmp
from ros_dmp.srv import *
from ros_dmp.msg import *
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose
import tf
import time
from interface_vision_utils.msg import ObjectPose

class GenerateMotionClass:
    def __init__(self):
        """ROS action server for generating motion using DMPs."""
        rospy.init_node("generate_motion_service_node")
        rospy.Service("generate_motion_service", GenerateMotion, self.generate_motion)
        rospy.loginfo("Started Motion Generation Service")

        # Publishers
        self.trajectory_pub = rospy.Publisher('~cartesian_trajectory', CartesianTrajectory, queue_size=1)
        self.path_pub = rospy.Publisher('~cartesian_path', PoseStamped, queue_size=1)

        # Parameters
        self.verbose = True
           
    def generate_motion(self, req):
        """Generates trajectory upon request in real-time."""
        rospy.loginfo("Received motion generation request")

        # Extract initial and goal poses
        initial_pose = np.array([
            req.initial_pose.pose.position.x, req.initial_pose.pose.position.y, req.initial_pose.pose.position.z,
            req.initial_pose.pose.orientation.x, req.initial_pose.pose.orientation.y, req.initial_pose.pose.orientation.z, req.initial_pose.pose.orientation.w
        ])

        goal_pose = np.array([
            req.goal_pose.pose.position.x, req.goal_pose.pose.position.y, req.goal_pose.pose.position.z,
            req.goal_pose.pose.orientation.x, req.goal_pose.pose.orientation.y, req.goal_pose.pose.orientation.z, req.goal_pose.pose.orientation.w
        ])

       
        # Create an object of class RollDmp (defined in the file 'roll_dmp.py')
        dmp = RollDmp(req.dmp_name, req.dt)

        # Record the start time for synchronization
        interval = req.dt
        start_time = time.time()

        # Initialize point counter
        point_counter = 0

        # Iterate over the generator and publish one point at a time, as soon as available
        for pos, _, _ in dmp.roll_generator(goal_pose, initial_pose, req.tau):

            # Build pose (with resepect to the base frame) and state messages
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos[:3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = pos[-4:]

            pose_pub = PoseStamped()
            pose_pub.header.stamp = rospy.Time.now()
            pose_pub.pose = pose
            pose_pub.header.frame_id = "dmp_ref"

            # Publish the path for visualization
            path = Path()
            path.header.frame_id = "dmp_ref"
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            path.poses.append(pose_stamped)
            self.path_pub.publish(pose_pub)

            # Increment point counter
            point_counter += 1

            # Sleep dynamically to maintain real-time execution
            elapsed_time = time.time() - start_time
            expected_time = point_counter * interval
            sleep_time = expected_time - elapsed_time
            if sleep_time > 0:
                rospy.sleep(sleep_time)

        # Prepare the response
        response = GenerateMotionResponse()
        response.result = "success"
        #response.cart_traj = cartesian_trajectory  # Only contains the last state
        return response




if __name__ == "__main__":
    try:
        motion_generator = GenerateMotionClass()
        rospy.spin()  # Keep the service node running
    except rospy.ROSInterruptException:
        rospy.loginfo("Generate Motion Service node terminated.")
