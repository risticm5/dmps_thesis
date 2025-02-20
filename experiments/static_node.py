#!/usr/bin/env python3
import rospy
import select
import sys
from interface_vision_utils.msg import ObjectPose
import numpy as np
import os
import csv

class ArucoPoseNode:
    def __init__(self):
        rospy.init_node('key_interrupt_node', anonymous=True)
        self.rate = rospy.Rate(10)  # Set loop rate (10Hz)
        self.aruco_pose = None
        self.complete_pose = []

        rospy.Subscriber("/object_pose", ObjectPose, self.object_pose_callback)

        rospy.loginfo("Press 'q' and Enter to stop the loop.")

    def object_pose_callback(self, msg):
        """Callback function to receive object pose messages."""
        if msg.name:  # Ensure the message contains at least one object
            self.aruco_pose = msg.pose[0]  # Assume the first pose for simplicity

    def check_for_keypress(self):
        """Check if a key was pressed (non-blocking)"""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            key = sys.stdin.read(1)
            return key
        return None

    def run(self):
        """Main loop that runs until 'q' is pressed and Enter is hit."""
        start_time = rospy.get_time()  # Get the initial timestamp

        while not rospy.is_shutdown():
            rospy.loginfo("Running task...")

            if self.aruco_pose:
                rospy.loginfo(self.aruco_pose)

                # Get current timestamp relative to start time
                timestamp = rospy.get_time() - start_time

                # Store pose data with timestamp
                current_pose = [
                    timestamp,
                    self.aruco_pose.translation.x,
                    self.aruco_pose.translation.y,
                    self.aruco_pose.translation.z,
                    self.aruco_pose.rotation.x,
                    self.aruco_pose.rotation.y,
                    self.aruco_pose.rotation.z,
                    self.aruco_pose.rotation.w,
                ]

                self.complete_pose.append(current_pose)

            # Check for key press
            key = self.check_for_keypress()
            if key == 'q':
                rospy.loginfo("Key 'q' pressed. Stopping the node.")
                self.save_to_csv()
                break
            
            self.rate.sleep()  # Control the loop rate

        rospy.loginfo("Node shutdown.")

    def save_to_csv(self):
        """Saves the collected pose data to a CSV file, including timestamps."""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "experiment6/static_test.csv")

        with open(file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header
            csv_writer.writerow(["Time (s)", "X", "Y", "Z", "Rot_X", "Rot_Y", "Rot_Z", "Rot_W"])
            
            # Write pose data
            csv_writer.writerows(self.complete_pose)  

        rospy.loginfo(f"Pose data saved to {file_path}")

if __name__ == "__main__":
    # Enable non-blocking input
    sys.stdin = open('/dev/tty')
    
    node = ArucoPoseNode()
    node.run()
