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

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        print(f"self.goal: {self.goal}")
        print(f"self.y0: {self.y0}")
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

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
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)

    def gen_coupling_terms(self, y_h, y_q):
        
        # Parameters for sigmoidal for distance and Ct
        a_d = -10
        delta_d = 0.35
        kt = 0

        # Get the robot pose
        self.tf_listener.waitForTransform("base_link", "dmp_link", rospy.Time(), rospy.Duration(1.0))
        (trans, _) = self.tf_listener.lookupTransform("base_link", "dmp_link", rospy.Time(0))

        # Initialize Cs
        Cs = np.zeros(7)
        ks = 1

        q_r = y_q # robot quaternion (dmp_link wrt base_link)
        q_h = y_h[-4:] # human quaternion (aruco wrt base_link)
        pom = np.linalg.norm(y_h[:3] - trans)
        print(f"The value of pom is: {pom}")

        # Compute Ct
        sigma_d = 1 / (1 + np.exp(a_d * (pom - delta_d)))
        Ct = kt * sigma_d

        # Compute quaternion error
        print(f"q_r: {q_r}")
        print(f"q_h: {q_h}")
        
        # Ensure q_r and q_h are formatted correctly for SciPy
        q_r_scipy = [q_r[0], q_r[1], q_r[2], q_r[3]]  # Convert from [w, x, y, z] to [x, y, z, w]
        q_h_scipy = [q_h[0], q_h[1], q_h[2], q_h[3]]  # Convert from [w, x, y, z] to [x, y, z, w]

        # Convert to Rotation objects
        q_r_rot = R.from_quat(q_r_scipy)  # q_r as Rotation object
        q_r_inv = q_r_rot.inv()  # Compute inverse
        q_h_rot = R.from_quat(q_h_scipy)  # q_h as Rotation object

        # Compute quaternion error (rotation difference)
        q_error = q_h_rot * q_r_inv  # Both are now Rotation objects
        q_error_quat = q_error.as_quat()  # Returns [x, y, z, w]

        # Convert quaternion back to Rotation
        r = R.from_quat(q_error_quat)
        #r = R.from_quat(q_error[1:]) 
        axis_angle = r.as_rotvec()  
        omega_error = 2 * axis_angle
        
        # Compute Cs
        print(f"omega_error: {omega_error}")
        print(f"sigma_d: {sigma_d}")
        
        omega_4x1 = np.hstack([omega_error, 0])
        Cs[-4:] = ks * sigma_d * omega_4x1
        print(f"Cs: {Cs}")

        return Ct, Cs
        

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
