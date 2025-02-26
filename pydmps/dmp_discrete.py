'''
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
'''
Modified by : Abhishek Padalkar
This softawre is modified version of original software provided by Travis DeWolf.
Modifications are made such that the software can be easily integrated in ROS.  
'''

from pydmps.dmp import DMPs
import numpy as np
import rospy
from interface_vision_utils.msg import ObjectPose
import tf
from scipy.spatial.transform import Rotation as R
import tf
import math

#Compute axis-angle rotation representation from quaternion (logarithmic map)
def qLog(q):
    # Ensure the quaternion is normalized
    norm_q = np.linalg.norm(q)
    if norm_q == 0:
        return R.from_quat([0, 0, 0, 1])
    q = q / norm_q
    # Convert quaternion to a rotation object
    rot = R.from_quat(q) # the right order of quat: qx,qy,qz,qw for scipy verson 1.10.1 that I use
    # Get the axis-angle representation (angle and axis) with :axis_angle = rot.as_rotvec(), if you need
    return rot

#Compute quaternion from axis-angle representation (exponential map)
def vecExp(axis_angle):
    # Create a rotation object from axis-angle
    rot = R.from_rotvec(axis_angle)
    # Convert to quaternion
    q = rot.as_quat()  # Returns [x, y, z, w]
    return q

# Computes quaternion gradient
def compute_velocity_optimized(q1, q0, dt):
    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q0 = q0 / np.linalg.norm(q0)
    # Convert quaternions to Rotation objects
    rot1 = R.from_quat(q1)
    rot0 = R.from_quat(q0)
    # Compute the relative rotation: q1 * q0^-1
    relative_rot = rot1 * rot0.inv()  # Quaternion multiplication and inversion
    # Get the axis-angle representation (rotation vector)
    axis_angle = relative_rot.as_rotvec()
    # Compute the velocities: 2 * axis_angle / dt
    velocities = 2 * axis_angle / dt
    return velocities

def compute_quaternion_distance(q2,q1):
    #computes rotational angles, that represent distance between quaternions q2 and q1

    # Compute axis-angle representations of quat
    rot1 = qLog(q1) #this will be rotation angles
    rot2 = qLog(q2)
    # Compute the relative rotation: q2 * q1^-1
    relative_rot = rot2 * rot1.inv() 
    axis_angle = relative_rot.as_rotvec()
    return axis_angle

class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):
        """
        """

        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern='discrete', **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax
        self.check_offset()

        # Initialize the listener
        self.tf_listener = tf.TransformListener()

        self.dist_vec = []

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        '''x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]'''

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num, pose):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        q1 = self.y0[3:]
        #q2 = self.goal[3:] #if scale factor is fixed
        #if scale factor changes 
        #changable_goal = np.array([0.0,0.0,0.0,1.0])
        changeable_goal = pose
        q2=changeable_goal
        distance = compute_quaternion_distance(q2,q1) #initial angular distance
        if dmp_num <=2:
            return x * (self.goal[dmp_num] - self.y0[dmp_num])
        else:
            return x * distance[dmp_num - 3]
        
    def gen_front_term_original(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        q1 = self.y0[3:]
        q2 = self.goal[3:] #if scale factor is fixed
        #if scale factor changes 
        #changable_goal = np.array([0.0,0.0,0.0,1.0])
        #q2=changable_goal
        distance = compute_quaternion_distance(q2,q1) #initial angular distance
        if dmp_num <=2:
            return x * (self.goal[dmp_num] - self.y0[dmp_num])
        else:
            return x * distance[dmp_num - 3]

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        q1 = self.y0[3:]
        q2 = self.goal[3:]
        distance = compute_quaternion_distance(q2,q1) #initial angular distance
        for d in range(self.n_dmps):
            # spatial scaling term
            if d<=2:
                k = (self.goal[d] - self.y0[d])
            else:
                # spatial scaling term -now should be computed differently for quaternions
                k = distance[d-3]   
            #print(f"spatial scaling term: {k}")
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)
        
    def gen_coupling_terms(self, y_h, y_r, goal, dy_r):
        
        # Parameters for sigmoidal for distance and Ct
        a_d = -50
        delta_d = 0.35
        kt = 0.0 # NOTE: k must range from 0 to 1
        

        # Get the robot pose
        #self.tf_listener.waitForTransform("base_link", "dmp_link", rospy.Time(), rospy.Duration(1.0))
        #(trans, _) = self.tf_listener.lookupTransform("base_link", "dmp_link", rospy.Time(0))
        q_r = y_r[-4:] # robot quaternion (dmp_link wrt base_link)
        q_h = y_h[-4:] # human quaternion (aruco wrt base_link)

        #print(f"The robot quaternion in euler angles is: {R.from_quat(q_r).as_euler('xyz', degrees=True)}")
        #print(f"The human quaternion in euler angles is: {R.from_quat(q_h).as_euler('xyz', degrees=True)}")
        #print(f"The robot quaternion in rotation vectors is: {R.from_quat(q_r).as_rotvec()*180/np.pi}")
        #print(f"The human quaternion in rotation vectors is: {R.from_quat(q_h).as_rotvec()*180/np.pi}")

        distance_pose = np.linalg.norm(y_h[:3] - y_r[:3])
        print(f"The distance in spatial coordinates is: {distance_pose}")
        self.dist_vec.append(distance_pose)
        # Compute Ct
        sigma_d = 1 / (1 + np.exp(a_d * (distance_pose - delta_d)))
        Ct = kt * sigma_d
        #Ct = kt

        # Initialize Cs
        Cs = np.zeros(6) #n_dmps=6
        ks = 0.3116 # Vary this in between 0 and 1
        ay = 25.0
        a_d = -10
        delta_d = 0.35 #now this represent mean for distance in rad

        velocities = dy_r[3:]

        # Compute Cs
        distance_angles = compute_quaternion_distance(q_h, q_r) #distance in rad
        print(f"The distance in rotation vectors is: {distance_angles}")
        print(f"The distance in degrees is: {distance_angles * 180 / np.pi}")
        distance_orientation = np.linalg.norm(distance_angles)
        print(f"The single parameter representing the rotation is {distance_orientation * 180 / np.pi}")  # Magnitude in degrees
        print(f"The singler param in radians is: {distance_orientation}")
        sigma_do = 1 / (1 + np.exp(a_d * (distance_orientation - delta_d)))
        print(f"The value of sigma_do is: {sigma_do}")
        print(f"THE VALUE OF TAUDYN IS: {self.tau_dyn}")
        Cs[3:] = ay * ks * velocities * sigma_do / self.tau_dyn 
        print(f"spatial copling term Cs is: {Cs}")

        # Define the fixed reference frame rotation in absolute terms
        rx = R.from_euler('x', 90, degrees=True)
        ry = R.from_euler('y', -90, degrees=True)
        fixed_ref_matrix = (rx * ry)  # Fixed frame rotation matrix

        # Convert human quaternion to rotation matrix
        human_rot_matrix = R.from_quat(q_h)

        # Compute the relative rotation: Human w.r.t Fixed frame
        relative_human_rotation = fixed_ref_matrix.inv() * human_rot_matrix
        # relative_human_rotation = human_rot_matrix.inv() * fixed_ref_matrix

        # Convert the relative rotation to a rotation vector
        distance_rotv = relative_human_rotation.as_rotvec()

        # Debugging: Print results
        print(f"The distance in rotation vectors is: {distance_rotv}")
        print(f"Distance scaling is: {distance_rotv * 180 / np.pi}")  # Convert to degrees
        print(f"The single parameter representing the rotation is {np.linalg.norm(distance_rotv) * 180 / np.pi}")  # Magnitude in degrees

                
        # Computation of Cm
        Cm = np.zeros(6)
        km = 3 # The smallest value is 1.0 (no amplification effect)
        by = ay / 4
        a_dm = -10
        delta_dm = 0.35 * np.ones(3) # 0.35 rad = 20 degrees; 0.7 rad = 40 degrees; 10 degrees = 0.17 rad
        theta = np.linalg.norm(distance_rotv)
        #sigma_dm = 1 / (1 + np.exp(a_dm * (theta - delta_dm))) # Value ranging from 0 to 1
        #print(f"The value of sigma_dm is: {sigma_dm}")
        #print(f"The amplification factor is: {km * sigma_dm}")
        #Cm[3:] = ay * by * km * sigma_dm * (distance_rotv / np.linalg.norm(distance_rotv))
        #Cm[3:] = km * sigma_dm * distance_rotv
        #Cm[3:] = km * distance_rotv
        #Cm[4] *= km # Amplify only the y component
        #print(f"The complete amplification term Cm is: {Cm}")

        relative_human_rotvec = relative_human_rotation.as_rotvec()
        # Sigmoidal
        sigma_dm = 1 + (km - 1) / (1 + np.exp(-a_dm * (relative_human_rotvec - delta_dm)))
        amplified_rotvec = sigma_dm * relative_human_rotvec
        print(f"The rleative rotations in terms of rotation angles is: {relative_human_rotvec}")
        print(f"The amplified relative rotations in terms of rotation angles is: {amplified_rotvec}")
        #print(f"The y component of the relative rotation is: {relative_human_rotvec[1]}")
        #amplified_rotvec[1] *= km # Amplify the rotation around y-axis
        amplified_human_rotation = R.from_rotvec(amplified_rotvec) # Realtive one
        #print(f"The amplified relative rotation in terms of euler angles is: {amplified_human_rotation.as_euler('xyz', degrees=True)}")
        amplified_goal_rotation = fixed_ref_matrix * amplified_human_rotation
        y_h_star = amplified_goal_rotation.as_quat()

        #print(f"THE human rotation in terms of euler angles is: {R.from_quat(y_h_star).as_euler('xyz', degrees=True)}")




        
    
        return Ct, Cs, Cm, y_h_star, self.dist_vec
    

# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test normal run
    dmp = DMPs_discrete(dt=.05, n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    y_track, dy_track, ddy_track = dmp.rollout()

    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track))*dmp.goal, 'r--', lw=2)
    plt.plot(y_track, lw=2)
    plt.title('DMP system - no forcing term')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['goal', 'system state'], loc='lower right')
    plt.tight_layout()

    # test imitation of path run
    plt.figure(2, figsize=(6, 4))
    n_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    path1 = np.sin(np.arange(0, 1, .01)*5)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.):] = .5
    
    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1, path2]))
        # change the scale of the movement
        dmp.goal[0] = 3
        dmp.goal[1] = 2

        y_track, dy_track, ddy_track = dmp.rollout()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(y_track[:, 0], lw=2)
        plt.subplot(212)
        plt.plot(y_track[:, 1], lw=2)

    plt.subplot(211)
    a = plt.plot(path1 / path1[-1] * dmp.goal[0], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend([a[0]], ['desired path'], loc='lower right')
    plt.subplot(212)
    b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['%i BFs' % i for i in n_bfs], loc='lower right')

    plt.tight_layout()
    plt.show()