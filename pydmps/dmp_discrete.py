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
import sys
import os
import csv


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

    def gen_coupling_terms(self, y_measured, goal, tau, y_dot, d0):
        '''
        Generates the coupling term
        y_dot float, array: current velocity array (6DOF)
        y_measured : the state feedback(measured) value (1DOF)
        goal : the goal position (6DOF)
        d0 : the initial distances(6DOF)
        '''
        #parameters: a_d, delta_d, a_dt, delta_dt, k_t, k_s, alfa_d
        #fixed parameters :
        alfa_d = 20.0 #second order filter parameter
        a_d= 13.0
        delta_d = -0.35
        #parameters to be optimized(in this case I took some initial values)
        k= 0.0 # k = 0: you do not have coupling terms
        a_dt = 1.0
        delta_dt = 0.29
        k_t=k_s=k
        #For now Cs is only implemneted for 1DOF
        y_d = y_dot[0]#taking one DOF, out of 6
        g = goal[0]
        print(f"The goal is: {g}")
        print(f"The measured value is: {y_measured[0]}")
        d = np.abs(g-y_measured[0]) #distance between new goal pose and current pose, just 1DOF
        #apply second order filter on the measured distance
        self.dd_de[0] = alfa_d*(alfa_d/4*(d-self.de[0])-self.d_de[0])
        print(f"The acceleration is: {self.dd_de}")
        self.d_de[0] += self.dd_de[0] * tau * self.dt
        print(f"The velocity is: {self.d_de}")
        self.de[0] += self.d_de[0] * self.dt
        pom=self.de[0]/d0[0]
        sigma_d = 1/(1+np.exp(-a_d*(pom+delta_d)))
        sigma_dt = 1/(1+np.exp(-a_dt*(self.d_de[0]+delta_dt)))
        Ct = k_t *sigma_d *sigma_dt
        Cs = -self.ay * tau *y_d * k_s *sigma_d *sigma_dt #for now 1DOF (y_d just first DOF)

        '''
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "distance.csv")  # File path in the script's directory
        with open(file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write the header
            #csv_writer.writerow(["distance_x (m)"])
            # Write the data to the file (flatten positions, velocities, accelerations)
            csv_writer.writerow(d)
        '''



        return Ct, Cs, d
        

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
