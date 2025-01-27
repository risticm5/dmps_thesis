#!/usr/bin/env python

import rospy
from interface_vision_utils.msg import ObjectPose  # Import your custom message type

def object_pose_callback(msg):
    """
    Callback function to process ObjectPose messages.
    """
    for i, name in enumerate(msg.name):
        # Check if the object is tracked
        is_tracked = msg.isTracked[i]
        pose = msg.pose[i]

        if is_tracked:
            rospy.loginfo(f"Object: {name}")
            rospy.loginfo(f"  Position: x={pose.translation.x}, y={pose.translation.y}, z={pose.translation.z}")
            rospy.loginfo(f"  Orientation: x={pose.rotation.x}, y={pose.rotation.y}, z={pose.rotation.z}, w={pose.rotation.w}")
        else:
            rospy.logwarn(f"Object: {name} is not tracked.")

def main():
    """
    Main function to initialize the node and subscribe to the /object_pose topic.
    """
    rospy.init_node('object_pose_subscriber', anonymous=True)

    # Subscribe to the /object_pose topic
    rospy.Subscriber("/object_pose", ObjectPose, object_pose_callback)

    rospy.loginfo("Subscribed to /object_pose. Waiting for messages...")
    rospy.spin()

if __name__ == '__main__':
    main()
