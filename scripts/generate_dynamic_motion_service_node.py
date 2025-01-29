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

        # Subscriber
        #self.aruco_pose = None
        #rospy.Subscriber("/object_pose", ObjectPose, self.object_pose_callback)

        self.verbose = True
           

    def generate_motion(self, req):
        """Generates trajectory upon request in real-time."""
        rospy.loginfo("Received motion generation request")

        # Extract initial and goal poses
        rpy_initial = tf.transformations.euler_from_quaternion([
            req.initial_pose.pose.orientation.x, req.initial_pose.pose.orientation.y,
            req.initial_pose.pose.orientation.z, req.initial_pose.pose.orientation.w
        ])
        initial_pose = np.array([
            req.initial_pose.pose.position.x, req.initial_pose.pose.position.y, req.initial_pose.pose.position.z,
            rpy_initial[0], rpy_initial[1], rpy_initial[2]
        ])

        rpy_goal = tf.transformations.euler_from_quaternion([
            req.goal_pose.pose.orientation.x, req.goal_pose.pose.orientation.y,
            req.goal_pose.pose.orientation.z, req.goal_pose.pose.orientation.w
        ])
        goal_pose = np.array([
            req.goal_pose.pose.position.x, req.goal_pose.pose.position.y, req.goal_pose.pose.position.z,
            rpy_goal[0], rpy_goal[1], rpy_goal[2]
        ])

        if self.verbose:
            rospy.loginfo(f"The initial pose is {initial_pose}")
            rospy.loginfo(f"The initial orientation in quaternion is {req.initial_pose.pose.orientation}")
            rospy.loginfo(f"The goal pose is {goal_pose}")
            rospy.loginfo(f"The final orientation in quaternion is {req.goal_pose.pose.orientation}")
        
        # Create an object of class RollDmp (defined in the file 'roll_dmp.py')
        dmp = RollDmp(req.dmp_name, req.dt)

        # Calculate the total time and expected interval per step
        #total_time = 1.0 / req.tau
        #num_points = int(dmp.dmp.timesteps / req.tau)  # Access timesteps from the DMPs_discrete instance
        #interval = total_time / num_points  # Time interval per point

        # Record the start time for synchronization
        interval = req.dt
        start_time = time.time()

        # Initialize point counter
        point_counter = 0

        # Iterate over the generator and publish one point at a time, as soon as available
        for pos, vel, acc in dmp.roll_generator(goal_pose, initial_pose, req.tau):

            # Build pose (with resepect to the base frame) and state messages
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos[:3]
            x, y, z, w = tf.transformations.quaternion_from_euler(*pos[3:])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w

            pose_pub = PoseStamped()
            pose_pub.header.stamp = rospy.Time.now()
            pose_pub.pose = pose
            pose_pub.header.frame_id = "dmp_ref"

            cartesian_state = CartesianState()
            cartesian_state.pose = pose
            cartesian_state.vel.linear.x, cartesian_state.vel.linear.y, cartesian_state.vel.linear.z = vel[:3]
            cartesian_state.vel.angular.x, cartesian_state.vel.angular.y, cartesian_state.vel.angular.z = vel[3:]
            cartesian_state.acc.linear.x, cartesian_state.acc.linear.y, cartesian_state.acc.linear.z = acc[:3]
            cartesian_state.acc.angular.x, cartesian_state.acc.angular.y, cartesian_state.acc.angular.z = acc[3:]

            # Publish only the current state
            cartesian_trajectory = CartesianTrajectory()
            cartesian_trajectory.header.frame_id = "dmp_ref"
            cartesian_trajectory.cartesian_state.append(cartesian_state)
            self.trajectory_pub.publish(cartesian_trajectory)

            # Publish the path for visualization
            path = Path()
            path.header.frame_id = "dmp_ref"
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            path.poses.append(pose_stamped)
            #self.path_pub.publish(path)
            self.path_pub.publish(pose_pub)

            # Log the currently published positions
            #rospy.loginfo(f"Published point {point_counter + 1}: x={pose.position.x:.4f}, y={pose.position.y:.4f}, z={pose.position.z:.4f}")

            # Increment point counter
            point_counter += 1

            # Sleep dynamically to maintain real-time execution
            elapsed_time = time.time() - start_time
            expected_time = point_counter * interval
            sleep_time = expected_time - elapsed_time
            if sleep_time > 0:
                rospy.sleep(sleep_time)

        # Record the end time and calculate total time taken
        end_time = time.time()
        total_time_taken = end_time - start_time

        # Log completion
        #rospy.loginfo("Real-time trajectory publication complete.")
        #rospy.loginfo(f"Total time taken to publish all points: {total_time_taken:.4f} seconds")
        #rospy.loginfo(f"Total number of points published: {point_counter}")

        # Prepare the response
        response = GenerateMotionResponse()
        response.result = "success"
        response.cart_traj = cartesian_trajectory  # Only contains the last state
        return response




if __name__ == "__main__":
    try:
        motion_generator = GenerateMotionClass()
        rospy.spin()  # Keep the service node running
    except rospy.ROSInterruptException:
        rospy.loginfo("Generate Motion Service node terminated.")
