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
#from threading import Lock

class DMPs(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(self, n_dmps, n_bfs, dt=.01,
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
        #self.aruco_pose_lock = Lock()
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps)*y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps)*goal
        self.goal = goal
        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

        self.ay = np.ones(n_dmps + 1) * 25. if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4. if by is None else by  # Schaal 2012
        self.x_old = 1

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        # set up the DMP system
        self.d0 = np.zeros(6)
        #self.tau_dyn = 1.0
        self.reset_state()

        self.reset_state_dynamic()

        # New variables
        #self.tau_dyn = 1.0
        self.d_de = np.zeros(6)
        self.d_not_filtered_plot = []
        self.d_filtered_plot = []
        self.goal_vec = []
        self.Ct_vec = []
        self.Cs_vec = []
        self.vel_vec = []
        #self.d_de = np.zeros(6)
        
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

    # y_measured is the object pose taken from camera
    # y_dot is the current velocity 
    def gen_coupling_terms(self, y_measured, goal, tau, y_dot, d0):
        raise NotImplementedError()
    
    def gen_weights(self, f_target):
        raise NotImplementedError()

    def imitate_path(self, y_des):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        """

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

        '''
        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d], kind='cubic', fill_value='extrapolate')
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path
        '''

        # Initialize path array
        path = np.zeros((self.n_dmps, self.timesteps))

        # Time points
        x_original = np.linspace(0, self.cs.run_time, y_des.shape[1])
        t_new = np.linspace(0, self.cs.run_time, self.timesteps)

        # Handle Cartesian data (x, y, z) - first 3 rows
        for d in range(3):  # Only process Cartesian dimensions directly
            path_gen = scipy.interpolate.interp1d(x_original, y_des[d], kind='cubic', fill_value="extrapolate")
            path[d] = path_gen(t_new)

        # Handle rotational data (roll, pitch, yaw) - last 3 rows
        euler_angles = y_des[3:6].T  # Extract and transpose for easier handling
        rotations = R.from_euler('xyz', euler_angles, degrees=False)  # Create Rotation objects

        # Convert to quaternions
        quaternions = rotations.as_quat()  # Shape: (N, 4)

        # Interpolate quaternions (use spherical linear interpolation)
        interp_func = scipy.interpolate.interp1d(x_original, quaternions, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_quaternions = interp_func(t_new)

        # Convert interpolated quaternions back to Euler angles
        interpolated_rotations = R.from_quat(interpolated_quaternions)
        interpolated_euler_angles = interpolated_rotations.as_euler('xyz', degrees=False).T

        # Assign interpolated rotational data back to the path array
        path[3:6] = interpolated_euler_angles

        # Update y_des with the interpolated path
        y_des = path

        # calculate velocity of y_des
        dy_des = np.diff(y_des) / self.dt
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self.n_dmps, 1)), dy_des))

        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self.n_dmps, 1)), ddy_des))

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = (ddy_des[d] - self.ay[d] *
                              (self.by[d] * (self.goal[d] - y_des[d]) -
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

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
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
            self.goal[:3] = goal[:3]

        # Define the cartesian initial trajectory (dmp_link wrt base_link)
        if y0[:3] is not None:
            self.y0[:3] = y0[:3]
        y = self.y0

        # Iniitalize robot quaternions (dmp_link wrt base_link) 
        self.y_q = self.euler_to_quaternion(self.y0[-3:])

        # Reset the state 
        self.reset_state_dynamic()
        
        # Start timing      
        start_time_global = time.time()

        while np.linalg.norm(y[:3] - self.goal[:3]) > 0.01:
            print(f"The error is {np.linalg.norm(y[:3] - self.goal[:3])}")

                       
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
            except AttributeError:
                rospy.logwarn("Incomplete Aruco pose data, skipping step.")
                rospy.sleep(self.dt)
                continue
            
            # Start timing
            start_time_step = time.time()

            # y, dy, ddy: pos, vel, acc of dmp_link wrt base_link
            y, dy, ddy = self.step(tau = tau, pose = current_pose, goal = self.goal, **kwargs)

            # End timing
            end_time_step = time.time()

            # Calculate and print the time taken for this step
            computation_time = end_time_step - start_time_step
            #print(f"Computation time for this step: {computation_time:.6f} seconds")

            elapsed_time_global = time.time() - start_time_global
            yield y, dy, ddy

        # The while loop is over: saving time!
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "distance.csv")  # File path in the script's directory
        with open(file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.d_not_filtered_plot)
            csv_writer.writerow(self.d_filtered_plot)
            csv_writer.writerow(self.Ct_vec)
            csv_writer.writerow(self.Cs_vec)
            csv_writer.writerow(self.vel_vec)

        



    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.y_old = self.y.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()
        self.x_old = 1

    def reset_state_dynamic(self):
        """Reset the system state for the dynamic dmps"""
        self.y = self.y0.copy()
        self.y_old = self.y.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()
        self.x_old = 1

        # New variables (filtered quantities)
        self.tau_dyn = 1.0
        self.y_q = self.euler_to_quaternion(self.y[-3:].copy())
        self.dy_q = np.zeros(4)
        self.ddy_q = np.zeros(4)

    def euler_to_quaternion(self, euler_angles):
        r = R.from_euler('xyz', euler_angles, degrees=False)  # Convert from Euler to Quaternion
        return r.as_quat()  # SciPy returns [x, y, z, w]

    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        Input: 
            q : array (4,) -> Quaternion [w, x, y, z]
        Output:
            euler : array (3,) -> [roll, pitch, yaw] in radians
        """
        print(f"The quaternion is: {q}")
        q_mod = [q[0], q[1], q[2], q[3]]
        r = R.from_quat(q_mod)  # Convert to [x, y, z, w] format
        euler_angles = r.as_euler('xyz', degrees=False)  # Convert to radians
        return euler_angles
    
    def compute_euler_velocity(self, euler, omega):

        print(f"The euler angles are: {euler}")
        print(f"omega is {omega}")
        roll, pitch, yaw = euler
        omega_xyz = omega[0:3]

        # Transformation matrix T(phi, theta)
        T = np.array([
            [1, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch)]
        ])

        euler_velocity = np.dot(T, omega_xyz)  # Transform to Euler rate
        return euler_velocity
    
    def quaternion_angular_velocity_to_euler(self, q, omega):
        """
        Convert angular velocity from quaternion representation to Euler angles.
        Inputs:
            q : array (4,) -> Quaternion [w, x, y, z]
            omega : array (3,) -> Angular velocity [wx, wy, wz] in quaternion frame
        Output:
            euler_velocity : array (3,) -> [roll_rate, pitch_rate, yaw_rate]
        """
        euler = self.quaternion_to_euler(q)
        roll, pitch, yaw = euler

        # Transformation matrix from body angular velocity to Euler rate
        T = np.array([
            [1, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch)]
        ])

        euler_velocity = np.dot(T, omega)  # Convert to Euler rate
        return euler_velocity

    
    def compute_euler_acceleration(self, euler, omega, omega_dot):
        roll, pitch, yaw = euler
        omega_xyz = omega[0:3]
        omega_dot_xyz = omega_dot[0:3]

        # Transformation matrix T(phi, theta)
        T = np.array([
            [1, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch)]
        ])

        # Compute time derivative of T(phi, theta)
        T_dot = np.array([
            [0, np.cos(roll) * np.tan(pitch) * omega_xyz[0] + np.sin(roll) / (np.cos(pitch)**2) * omega_xyz[1], -np.sin(roll) * np.tan(pitch) * omega_xyz[0] + np.cos(roll) / (np.cos(pitch)**2) * omega_xyz[1]],
            [0, -np.sin(roll) * omega_xyz[0], -np.cos(roll) * omega_xyz[0]],
            [0, (np.cos(roll) * omega_xyz[0] / np.cos(pitch)) + (np.sin(roll) * np.tan(pitch) * omega_xyz[1] / np.cos(pitch)), (-np.sin(roll) * omega_xyz[0] / np.cos(pitch)) + (np.cos(roll) * np.tan(pitch) * omega_xyz[1] / np.cos(pitch))]
        ])

        # Compute Euler acceleration
        euler_acceleration = np.dot(T, omega_dot_xyz) + np.dot(T_dot, omega_xyz)
        return euler_acceleration

    def step(self, tau=1.0, error=0.0, external_force=None, pose=None, goal=None):
        ''' 
        tau = tau0 defined with PBO
        pose = human hand wrt base_link (pose[-4:] is the dynamic goal)
        goal: final x, y, z coordinates of dmp_link wrt base_link
        '''
        error_coupling = 1.0 / (1.0 + error)

        Ct, Cs = self.gen_coupling_terms(pose, self.y_q)

        # Save some values for the plots
        #self.d_not_filtered_plot.append(d_plot[0])
        #self.d_filtered_plot.append(self.de[0])
        #self.goal_vec.append(goal[0])
        #self.Cs_vec.append(Cs[0])
        #self.Ct_vec.append(Ct)
        #self.vel_vec.append(vel_filtered_plot[0])

        # Compute the new tau
        tau0 = tau
        self.tau_dyn = tau0 * (1 - Ct)
        print(f"The value of tau_dyn is: {self.tau_dyn}")

        # compute phase and basis functions
        x = self.cs.step(tau=self.tau_dyn, error_coupling=error_coupling)
        psi = self.gen_psi(x)
        f_vec = []
        f_q = []
        # precompute forcing terms
        for d in range(self.n_dmps):
            f = (self.gen_front_term(x, d) *
                 (np.dot(psi, self.w[d])) / np.sum(psi))
            f_vec.append(f)
        
        # Convert to forcing terms for quaternions
        #q = R.from_euler('xyz', f_vec[-3:], degrees=True).as_quat()  # [x, y, z, w]
        f_q = R.from_rotvec(f_vec[-3:]).as_quat()

        # Scale the quaternion forcing term (approximation)
        f_quaternion = np.concatenate(([0], f_q[0:3]))


        # Loop from d = 1 to d = 6
        for d in range(self.n_dmps + 1):
            print(f"the value of d is: {d}")
            # generate the forcing term
            #f = (self.gen_front_term(x, d) *
            #     (np.dot(psi, self.w[d])) / np.sum(psi))
            #print(f"The value of the forcing term f is: {f}")
            if d <= 2:
                
                # DMP acceleration
                self.ddy[d] = (self.ay[d] *
                            (self.by[d] * (self.goal[d] - self.y[d]) -
                            self.dy[d]/self.tau_dyn) + f_vec[d]  + Cs[d]) * self.tau_dyn
                
                if external_force is not None:
                    self.ddy[d] += external_force[d]

                self.dy[d] += self.ddy[d] * self.tau_dyn * self.dt * error_coupling
                self.y[d] += self.dy[d] * self.dt * error_coupling

            else:

                # Equations in terms of quaternions (d = 3, 4, 5, 6)
                self.ddy_q[d - 3] = (self.ay[d] *
                            (self.by[d] * (pose[d] - self.y_q[d - 3]) -
                            self.dy_q[d - 3]/self.tau_dyn) + f_quaternion[d - 3] + Cs[d]) * self.tau_dyn
                
                if external_force is not None:
                    self.ddy_q[d - 3] += external_force[d]

                self.dy_q[d - 3] += self.ddy_q[d - 3] * self.tau_dyn * self.dt * error_coupling
                self.y_q[d - 3] += self.dy_q[d - 3] * self.dt * error_coupling

        # Convert from quaternion to euler
        q_r = self.y_q[-4:]
        omega_r = self.dy_q[-4:]
        omega_r_dot = self.ddy_q[-4:]

        print(f"q_r is: {q_r}")

        # Convert quaternion to Euler angles
        euler_angles = self.quaternion_to_euler(q_r)
        euler_velocity = self.compute_euler_velocity(euler_angles, omega_r)
        euler_acceleration = self.compute_euler_acceleration(euler_angles, omega_r, omega_r_dot)

        # Append the euler angles to the y vector
        self.y[-3:] = euler_angles
        self.dy[-3:] = euler_velocity
        self.ddy[-3:] = euler_acceleration

        # 6x1 vectors representing the trajectories of dmp_link wrt base_link
        return self.y, self.dy, self.ddy

    
    def step_original(self, tau=None, error=0.0, external_force=None, pose=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                increase tau to make the system execute faster
        error float: optional system feedback
        """
        print(f"The value of tau is {tau}")
        #rospy.init_node("object_pose_subscriber")
        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)
        print(f"The value of the phase variable x is: {x}")

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):

            # Here some modifications could be done to
            # Account for coupling terms based on the distance...

            # generate the forcing term
            f = (self.gen_front_term(x, d) *
                (np.dot(psi, self.w[d])) / np.sum(psi))
            print(f"The value of the forcing term f is: {f}")

            # DMP acceleration
            self.ddy[d] = (self.ay[d] *
                        (self.by[d] * (self.goal[d] - self.y[d]) -
                        self.dy[d]/tau) + f) * tau
            if external_force is not None:
                self.ddy[d] += external_force[d]

            # Compute velocity and position
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * self.dt * error_coupling


        return self.y, self.dy, self.ddy
