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

class GenerateMotionClass:
    def __init__(self):
        """ROS action server for generating motion using DMPs."""
        rospy.init_node("generate_motion_service_node")
        rospy.Service("generate_motion_service", GenerateMotion, self.generate_motion)
        rospy.loginfo("Started Motion Generation Service")

        # Publishers
        self.trajectory_pub = rospy.Publisher('~cartesian_trajectory', CartesianTrajectory, queue_size=1)
        self.path_pub = rospy.Publisher('~cartesian_path', Path, queue_size=1)

        self.verbose = True

    def generate_motion(self, req):
        """Generates trajectory upon request."""
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
            rospy.loginfo(f"The goal pose is {goal_pose}")

        dmp = RollDmp(req.dmp_name, req.dt)

        # Dynamically compute and publish points during rollout
        cartesian_trajectory = CartesianTrajectory()
        cartesian_trajectory.header.frame_id = "dmp_ref"
        path = Path()
        path.header.frame_id = "dmp_ref"

        rospy.loginfo("Publishing points in real-time...")

        # Generate the trajectory points first to determine the total number
        trajectory_points = list(dmp.roll_generator(goal_pose, initial_pose, req.tau))
        num_points = len(trajectory_points)

        # Calculate the interval based on tau and the number of points
        interval = (1.0 / req.tau) / num_points  # Time interval per point in seconds

        # Start publishing trajectory
        start_time = time.time()

        # Record the start time for the entire trajectory execution
        start_execution_time = time.time()
        counter = 1

        for pos, vel, acc in trajectory_points:
            # Build pose and state messages
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos[:3]
            x, y, z, w = tf.transformations.quaternion_from_euler(*pos[3:])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w

            cartesian_state = CartesianState()
            cartesian_state.pose = pose
            cartesian_state.vel.linear.x, cartesian_state.vel.linear.y, cartesian_state.vel.linear.z = vel[:3]
            cartesian_state.vel.angular.x, cartesian_state.vel.angular.y, cartesian_state.vel.angular.z = vel[3:]
            cartesian_state.acc.linear.x, cartesian_state.acc.linear.y, cartesian_state.acc.linear.z = acc[:3]
            cartesian_state.acc.angular.x, cartesian_state.acc.angular.y, cartesian_state.acc.angular.z = acc[3:]

            # cartesian_trajectory.cartesian_state.append(cartesian_state)
            # Create a new CartesianTrajectory message with only the current state
            single_state_trajectory = CartesianTrajectory()
            single_state_trajectory.header.frame_id = "dmp_ref"
            single_state_trajectory.header.stamp = rospy.Time.now()  # Add a timestamp
            single_state_trajectory.cartesian_state = [cartesian_state]  # Add only the current state

            # Publish the current trajectory point
            self.trajectory_pub.publish(single_state_trajectory)


            # Publish trajectory point
            #self.trajectory_pub.publish(cartesian_trajectory)

            # Publish path
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            path.poses.append(pose_stamped)
            self.path_pub.publish(path)

            # Sleep for the calculated interval to maintain real-time execution
            elapsed_time = time.time() - start_time
            sleep_time = interval - elapsed_time
            if sleep_time > 0:
                rospy.sleep(sleep_time)
            start_time = time.time()  # Update start time for the next point

            # Update counter
            counter += 1
            

        # Record the end time for the entire trajectory execution
        end_execution_time = time.time()

        # Calculate the total time taken
        total_time = end_execution_time - start_execution_time

        # Log or print the total time
        rospy.loginfo(f"Total time taken to execute the trajectory: {total_time:.4f} seconds")
        rospy.loginfo(f"Total number of points published: {counter}")

        rospy.loginfo("Real-time trajectory publication complete.")
        response = GenerateMotionResponse()
        response.result = "success"
        response.cart_traj = cartesian_trajectory
        return response


if __name__ == "__main__":
    try:
        motion_generator = GenerateMotionClass()
        rospy.spin()  # Keep the service node running
    except rospy.ROSInterruptException:
        rospy.loginfo("Generate Motion Service node terminated.")
