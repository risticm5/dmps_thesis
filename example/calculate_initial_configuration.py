#!/usr/bin/env python3
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import tf

''' 
NOTE 1: run this script after launching in another terminal:
roslaunch ur5e_mics_bringup fake_start.launch

One possible outcome is the following:
Joint values: (-1.300555572858762, -1.6577401740289908, -1.979181216787131, -1.0779634984505604, 1.5712722377047859, 0.269147222597415)
Names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

NOTE 2; the idea is that you copy here the values imposed in the starting configuration in the
file 'motion_generation_client.py' in the 'initial_pose' variable, and then impose the joint values in the
file 'taw_controller.launch'

'''

rospy.init_node("ik_client")

rospy.wait_for_service("/compute_ik")
compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)

# Define the desired Cartesian pose
request = GetPositionIKRequest()
request.ik_request.group_name = "ur5e"  # Specify your robot's planning group
request.ik_request.pose_stamped.header.frame_id = "base_link"
request.ik_request.pose_stamped.pose.position.x = 0.0
request.ik_request.pose_stamped.pose.position.y = 0.5
request.ik_request.pose_stamped.pose.position.z = 0.3
x1, y1, z1, w1 = tf.transformations.quaternion_from_euler(3.14, 0.0, 0.0) # rx, ry, rz
request.ik_request.pose_stamped.pose.orientation.x = x1
request.ik_request.pose_stamped.pose.orientation.y = y1
request.ik_request.pose_stamped.pose.orientation.z = z1
request.ik_request.pose_stamped.pose.orientation.w = w1
request.ik_request.avoid_collisions = False

try:
    response = compute_ik(request)
    print("Joint values:", response.solution.joint_state.position)
    print(f"Names: {response.solution.joint_state.name}")
except rospy.ServiceException as e:
    print("Service call failed:", e)
