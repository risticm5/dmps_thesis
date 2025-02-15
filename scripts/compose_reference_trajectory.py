#!/usr/bin/env python
import rospy
import tf
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool  # Replace with appropriate message type for Oculus button A
from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback
import csv
import os
import sys

class OculusPoseLogger:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('oculus_pose_logger', anonymous=True)

        # TF listener for transformations
        self.tf_listener = tf.TransformListener()

        # Variable to track button A state
        self.button_a_pressed = False

        # Open the CSV file
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(self.script_dir)
        self.file_path = os.path.join(self.script_dir, "../reference_trajectory_random.csv")

        # Open the file and keep it open for the lifetime of the object
        self.csv_file = open(self.file_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

        # Variable to save the pose
        self.save_values = False

        # Subscribe to button A topic
        rospy.Subscriber("/q2r_right_hand_inputs", OVR2ROSInputs, self.button_a_callback, queue_size=1)

        # Run the node
        self.run()

    def button_a_callback(self, msg):
        """Callback for button A state."""
        if msg.button_lower:
            self.save_values = True
        else:
            self.save_values = False

    def get_tool_pose(self):
        """Get the pose of tool0 with respect to base_link."""
        try:
            # Wait for the transform to be available
            self.tf_listener.waitForTransform("base_link", "dmp_link", rospy.Time(), rospy.Duration(1.0))
            
            # Lookup the transform
            (trans, rot) = self.tf_listener.lookupTransform("base_link", "dmp_link", rospy.Time(0))
            return trans, rot
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Error getting transform: {e}")
            return None, None

    def log_pose(self):
        """Log the pose to the terminal and CSV file."""
        trans, rot = self.get_tool_pose()
        if trans and rot:
            # Get current time
            timestamp = rospy.Time.now().to_sec()

            # Print pose
            rospy.loginfo(f"Time: {timestamp}, Position: {trans}, Orientation: {rot}")

            # Write to CSV
            self.csv_writer.writerow([timestamp, *trans, *rot])

    def run(self):
        """Main loop of the node."""
        rate = rospy.Rate(50)  # 100 Hz
        rospy.loginfo("Oculus Pose Logger Node is running. Waiting for button A trigger...")
        
        try:
            while not rospy.is_shutdown():
                if self.save_values:
                    self.log_pose()
                rate.sleep()
        except rospy.ROSInterruptException:
            rospy.loginfo("Shutting down Oculus Pose Logger Node.")
        finally:
            # Close the CSV file when the node shuts down
            self.csv_file.close()

if __name__ == "__main__":
    OculusPoseLogger()
