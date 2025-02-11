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
import numpy as np
import rospy
from pydmps.cs import CanonicalSystem
from interface_vision_utils.msg import ObjectPose
import time
import os
import csv
from scipy.spatial.transform import Rotation as R


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


def compute_quaternion_distance(q2,q1):
    #computes rotational angles, that represent distance between quaternions q2 and q1

    # Compute axis-angle representations of quat
    rot1 = qLog(q1) #this will be rotation angles
    rot2 = qLog(q2)
    # Compute the relative rotation: q2 * q1^-1
    relative_rot = rot2 * rot1.inv() 
    axis_angle = relative_rot.as_rotvec()
    return axis_angle


class DMPs(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(self, n_dmps, n_bfs, dt=.02, #in order to have dt equal to teleoperation dt
                 y0=0, goal=1, w=None,
                 ay=None, by=None, **kwargs):
        """
        n_dmps int: number of dynamic motor primitives
        n_bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """
        learning_mode = False
        if "learning_mode" in kwargs:
            learning_mode = kwargs["learning_mode"]
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        self.aruco_pose = None
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps+1)*y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps+1)*goal
        self.goal = goal
        #self.goal[-4:] = [0, 0, 0, 1]  # Initialize quaternions to identity
        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

        self.ay = np.ones(n_dmps) * 25. if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4. if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        # set up the DMP system
        self.d0 = np.zeros(7)
        self.reset_state()
        self.reset_state_dynamic()
        
        # New variables
        self.Ct_vec = []
        self.Cs_vec = []
        self.vel_vec = []

        rospy.Subscriber("/object_pose", ObjectPose, self.object_pose_callback)

    def object_pose_callback(self, msg):
        """
        Callback function to process ObjectPose messages.
        """
        for i, name in enumerate(msg.name):
            # Check if the object is tracked
            is_tracked = msg.isTracked[i]
            self.aruco_pose = msg.pose[i]

            '''
            if is_tracked:
                rospy.loginfo(f"Object: {name}")
                rospy.loginfo(f"  Position: x={pose.translation.x}, y={pose.translation.y}, z={pose.translation.z}")
                rospy.loginfo(f"  Orientation: x={pose.rotation.x}, y={pose.rotation.y}, z={pose.rotation.z}, w={pose.rotation.w}")
            else:
                rospy.logwarn(f"Object: {name} is not tracked.")

            '''

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if (self.y0[d] == self.goal[d]):
                self.goal[d] += 1e-4

    def gen_front_term(self, x, dmp_num):
        raise NotImplementedError()

    def gen_goal(self, y_des):
        raise NotImplementedError()

    def gen_psi(self):
        raise NotImplementedError()

    def gen_weights(self, f_target):
        raise NotImplementedError()

    # y_measured is the object pose taken from camera
    # y_dot is the current velocity 
    def gen_coupling_terms(self, y_measured, goal, tau, y_dot, d0):
        raise NotImplementedError()
    
    def imitate_path(self, y_des):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        """
        #There are some changes added by Christian in order to have normal interpolation!
        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        #self.goal = self.gen_goal(y_des)
        self.goal = y_des[:, -1].copy()

        self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate
        from scipy.spatial.transform import Rotation as R

        # Initialize path array
        path = np.zeros((self.n_dmps+1, self.timesteps))

        # Time points
        x_original = np.linspace(0, self.cs.run_time, y_des.shape[1])
        t_new = np.linspace(0, self.cs.run_time, self.timesteps)

        # Handle Cartesian data (x, y, z) - first 3 rows
        for d in range(3):  # Only process Cartesian dimensions directly
            path_gen = scipy.interpolate.interp1d(x_original, y_des[d], kind='cubic', fill_value="extrapolate")
            path[d] = path_gen(t_new)

        # Handle rotational data (quaternions)
        quaternions = y_des[3:7].T 

        # Interpolate quaternions (use spherical linear interpolation)
        interp_func = scipy.interpolate.interp1d(x_original, quaternions, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_quaternions = interp_func(t_new)

        path[3:self.n_dmps+1] = interpolated_quaternions.T

        # Update y_des with the interpolated path
        y_des = path

        #computaion of velocities is now different for quaternions
        dy_des = np.zeros((self.n_dmps, self.timesteps)) #velocities
        dy_des[:,0] = 0.0 # add zero to the beginning of every row
        for i in range(1,self.timesteps):
            dy_des[0,i] = (y_des[0,i] - y_des[0,i-1])/self.dt
            dy_des[1,i] = (y_des[1,i] - y_des[1,i-1])/self.dt
            dy_des[2,i] = (y_des[2,i] - y_des[2,i-1])/self.dt
            q2 = y_des[3:,i]
            q1 = y_des[3:,i-1] 
            dy_des[3:,i] = compute_quaternion_distance(q2,q1) / self.dt #no more multiplication with 2 (2*logq) 
        
        # calculate acceleration of y_des - same for all variables
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self.n_dmps, 1)), ddy_des))

        # find the force required to move along this trajectory
        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        #we need to precompute distances
        distances = np.zeros((3,self.timesteps))
        for i in range(self.timesteps):
            q1 = y_des[3:,i]
            q2 = self.goal[3:]
            distances[:,i] = compute_quaternion_distance(q2,q1) #no more multiplication with 2 (2*logq)

        for d in range(self.n_dmps):
            if d <= 2:
                f_target[:, d] = (ddy_des[d] - self.ay[d] *
                                (self.by[d] * (self.goal[d] - y_des[d]) -
                                dy_des[d]))
            #computation for quaternions -> all integrated
            else: 
                f_target[:,d] = (ddy_des[d] - self.ay[d] *
                                (self.by[d] * distances[d-3] -
                                dy_des[d]))
        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        self.reset_state()
        return self.w


    def rollout(self, timesteps=None, goal=None, y0=None, **kwargs):
        """Generate a system trial, no feedback is incorporated."""

        if goal is not None:
            self.goal = goal

        if y0 is not None:
            self.y0 = y0

        self.reset_state()

        if timesteps is None:
            if 'tau' in kwargs:
                timesteps = int(self.timesteps / kwargs['tau'])
            else:
                timesteps = self.timesteps

        #change of ay and by parameters - GLISp optimiz of this param
        '''
        if 'ay_glisp' in kwargs:
            self.ay=np.ones(self.n_dmps) * kwargs['ay_glisp']
        if 'by_glisp' in kwargs:
            self.by=np.ones(self.n_dmps) * kwargs['by_glisp']
        '''
        
        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps + 1))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))
        for t in range(timesteps):

            # run and record timestep (you are 'appending' values)
            y_track[t], dy_track[t], ddy_track[t] = self.step_original(**kwargs)

            # At this point you can even think of publishing the current value...

        return y_track, dy_track, ddy_track
    
    def roll_generator(self, goal=None, y0=None, tau=1.0, **kwargs):
        # Define the cartesian goal (dmp_link wrt base_link)
        if goal is not None:
            self.goal = goal

        # Define the cartesian initial trajectory (dmp_link wrt base_link)
        if y0 is not None:
            self.y0 = y0
        y = self.y0

        # Reset the state 
        self.reset_state_dynamic()
        iteration = 0
        while np.linalg.norm(y[:3] - self.goal[:3]) > 0.01:
            #print(f"The error is {np.linalg.norm(y[:3] - self.goal[:3])}")
            #if we want to add also the condition on orientations
            #q1 = self.y[3:]
            #q2 = self.goal[3:] #q2 = pose[3:]
            #distance = compute_quaternion_distance(q2,q1)
            #distance_degrees = distance*180/np.pi
            #print(f"The angle error in degrees is {np.linalg.norm(distance_degrees)}")
            
            
            if self.aruco_pose is None:
                rospy.logwarn("Aruco pose is not yet available, skipping step.")
                rospy.sleep(self.dt)
                continue
            # Get the human pose (aruco marker) with respect to the base_link
            try:
                
                current_pose = np.array([
                    self.aruco_pose.translation.x,
                    self.aruco_pose.translation.y,
                    self.aruco_pose.translation.z,
                    self.aruco_pose.rotation.x,
                    self.aruco_pose.rotation.y,
                    self.aruco_pose.rotation.z,
                    self.aruco_pose.rotation.w,
                ])
                
                #current_pose = np.array([0, 0, 0, 0, 0, 0, 1])
            except AttributeError:
                rospy.logwarn("Incomplete Aruco pose data, skipping step.")
                rospy.sleep(self.dt)
                continue
            
                
            
                
            
            

            
            #since I dont have camera I will assume fixed orientation(info about camera pose is not used now)
            #current_pose = np.array([0.0, 0.0, 0.0, 0.49780278645538634, -0.5205946938151316, -0.5117137489328473, 0.4683188974639753])
            #current_pose = self.goal
            #current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            '''
            r = R.from_euler('z', 90, degrees=True)
            q = r.as_quat()
            current_pose[3:] = q

            if iteration > 35:
                r1 = R.from_euler('x', -45, degrees=True)
                new_rot = r1 * r
                q1 = new_rot.as_quat()
                angle = new_rot.as_rotvec()
                print(f"The new angle in degrees is {angle*180/np.pi}")
                current_pose[3:] = q1
            '''

            
            
            
            
            # Start timing
            start_time_step = time.time()
            iteration = iteration+1
            # y, dy, ddy: pos, vel, acc of dmp_link wrt base_link
            y, dy, ddy = self.step(tau = tau, pose = current_pose, goal = self.goal, **kwargs)
            yield y, dy, ddy

        # The while loop is over: saving time!
        '''
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "distance.csv")  # File path in the script's directory
        with open(file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.d_not_filtered_plot)
            csv_writer.writerow(self.d_filtered_plot)
            csv_writer.writerow(self.Ct_vec)
            csv_writer.writerow(self.Cs_vec)
            csv_writer.writerow(self.vel_vec)
        '''

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def reset_state_dynamic(self):
        """Reset the system state for the dynamic dmps"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()
        self.tau_dyn = 1.0

    def step(self, tau=1.0, error=0.0, external_force=None, pose=None, goal=None):
        '''
        tau = tau0 defined with PBO
        pose = human hand wrt base_link (pose[-4:] is the dynamic goal)
        goal: final x, y, z coordinates of dmp_link wrt base_link
        '''
        error_coupling = 1.0 / (1.0 + error)
        Ct, Cs = self.gen_coupling_terms(pose, self.y,self.goal,self.dy)
        # Compute the new tau
        tau0 = tau
        self.tau_dyn = tau0 * (1 - Ct)
        print(f"The value of tau_dyn is: {self.tau_dyn}")
        # compute phase and basis functions
        x = self.cs.step(tau=self.tau_dyn, error_coupling=error_coupling)
        psi = self.gen_psi(x)
        print(f"The value of x is: {x}")
        #precompute quaternion distance : diference betwwen current orientation and goal orientation
        q1 = self.y[3:]
        #if the goal orientation is fixed
        #q2 = self.goal[3:]
        #if the goal orientation is changable
        q2 = pose[3:]
        distance = compute_quaternion_distance(q2,q1)
        #print(f”The value of distance is: {distance*180/np.pi}“)
        for d in range(self.n_dmps):
            # Solve the equations
            f = (self.gen_front_term(x, d, q2) *
                 (np.dot(psi, self.w[d])) / np.sum(psi))
            if d <= 2:
                self.ddy[d] = (self.ay[d] *
                            (self.by[d] * (self.goal[d] - self.y[d]) -
                            self.dy[d]/self.tau_dyn) + f  + Cs[d]) * (self.tau_dyn ** 2)
                if external_force is not None:
                    self.ddy[d] += external_force[d]
                self.dy[d] += self.ddy[d] * self.dt * error_coupling
                self.y[d] += self.dy[d] * self.dt * error_coupling
            else:
                # Equations in terms of quaternions (d = 3, 4, 5)
                self.ddy[d] = (self.ay[d] *
                            (self.by[d] * distance[d-3] -
                            self.dy[d]/self.tau_dyn) + f + Cs[d]) * (self.tau_dyn ** 2)
                #print(f”The value of f_target[d] is: {f}“)
                #print(f”The value of distances[d] is: {distance[d-3]}“)
                #print(f”The angle error in degrees is {np.linalg.norm(distance[d-3]*180/np.pi)}“)
                #total = self.by[d] * distance[d-3] - self.dy[d]/self.tau_dyn
                #part_one = self.by[d] * distance[d-3]
                #part_two = - self.dy[d]/self.tau_dyn
                #print(f”The value of f_target[d] is: {f}“)
                #print(f”The value of total is: {total}“)
                #print(f”The value of part one is: {part_one}“)
                #print(f”The value of part two is: {part_two}“)
                if external_force is not None:
                    self.ddy[d] += external_force[d]
                self.dy[d] += self.ddy[d] * self.dt * error_coupling #velocity computed same
        #I need to have all components of velocities in order to compute new quaternion orientation
        #I need to compute rotation anlges in order to execute quaternion multiplication!
        rot1 = R.from_rotvec(self.dt * error_coupling * self.dy[3:]) #no more multiplicat with 2
        rot2 = R.from_quat([self.y[3], self.y[4], self.y[5], self.y[6]])
        new_rot = rot1 * rot2
        y_angles = new_rot.as_rotvec()
        self.y[3:] = new_rot.as_quat()
        print(f"The angle values of y are: {y_angles*180/np.pi}")
        # 7x1 vectors representing the trajectories of dmp_link wrt base_link
        #print(f”The value of y is: {self.y}“)
        #print(f”The value of y_d is: {self.dy}“)
        #print(f”The value of y_dd is: {self.ddy}“)
        return self.y, self.dy, self.ddy   
    

    def step_original(self, tau=1.0, error=0.0, external_force=None, goal=None):
        ''' 
        tau = tau0 defined with PBO
        pose = human hand wrt base_link (pose[-4:] is the dynamic goal)
        goal: final x, y, z coordinates of dmp_link wrt base_link
        '''
        error_coupling = 1.0 / (1.0 + error)

        #Ct, Cs = self.gen_coupling_terms(pose, self.y,self.goal,self.dy)

        # Compute the new tau
        #tau0 = tau
        #self.tau_dyn = tau0 * (1 - Ct)
        #print(f"The value of tau_dyn is: {self.tau_dyn}")

        # compute phase and basis functions
        x = self.cs.step(tau=tau, error_coupling=error_coupling)
        psi = self.gen_psi(x)
        print(f"The value of x is: {x}")

        #precompute quaternion distance : diference betwwen current orientation and goal orientation
        q1 = self.y[3:]
        #if the goal orientation is fixed
        #q2 = self.goal[3:] 
        #if the goal orientation is changable
        q2 = self.goal[3:]
        distance = compute_quaternion_distance(q2,q1)
        #print(f"The value of distance is: {distance*180/np.pi}")

        for d in range(self.n_dmps):
            # Solve the equations
            f = (self.gen_front_term_original(x, d) *
                 (np.dot(psi, self.w[d])) / np.sum(psi))
            if d <= 2:
                self.ddy[d] = (self.ay[d] *
                            (self.by[d] * (self.goal[d] - self.y[d]) -
                            self.dy[d]/tau) + f) * (tau ** 2)
                
                if external_force is not None:
                    self.ddy[d] += external_force[d]

                self.dy[d] += self.ddy[d] * self.dt * error_coupling
                self.y[d] += self.dy[d] * self.dt * error_coupling
            else:
                # Equations in terms of quaternions (d = 3, 4, 5)
                self.ddy[d] = (self.ay[d] *
                            (self.by[d] * distance[d-3] -
                            self.dy[d]/tau) + f) * (tau ** 2)
                #print(f"The value of f_target[d] is: {f}")
                #print(f"The value of distances[d] is: {distance[d-3]}")
                #print(f"The angle error in degrees is {np.linalg.norm(distance[d-3]*180/np.pi)}")
                #total = self.by[d] * distance[d-3] - self.dy[d]/self.tau_dyn
                #part_one = self.by[d] * distance[d-3]
                #part_two = - self.dy[d]/self.tau_dyn

                #print(f"The value of f_target[d] is: {f}")
                #print(f"The value of total is: {total}")
                #print(f"The value of part one is: {part_one}")
                #print(f"The value of part two is: {part_two}")
                
                if external_force is not None:
                    self.ddy[d] += external_force[d]

                self.dy[d] += self.ddy[d] * self.dt * error_coupling #velocity computed same
                
        #I need to have all components of velocities in order to compute new quaternion orientation
        #I need to compute rotation anlges in order to execute quaternion multiplication!
        rot1 = R.from_rotvec(self.dt * error_coupling * self.dy[3:]) #no more multiplicat with 2
        rot2 = R.from_quat([self.y[3], self.y[4], self.y[5], self.y[6]])
        new_rot = rot1 * rot2
        y_angles = new_rot.as_rotvec()
        self.y[3:] = new_rot.as_quat()
        #print(f"The angle values of y are: {y_angles*180/np.pi}")
        # 7x1 vectors representing the trajectories of dmp_link wrt base_link
        #print(f"The value of y is: {self.y}")
        #print(f"The value of y_d is: {self.dy}")
        #print(f"The value of y_dd is: {self.ddy}")
        return self.y, self.dy, self.ddy