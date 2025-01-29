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

        self.ay = np.ones(n_dmps) * 25. if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4. if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        # set up the DMP system
        self.reset_state()

        # Subscriber
        #rospy.init_node('object_pose_subscriber', anonymous=True)
        
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
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

            # At this point you can even think of publishing the current value...

        return y_track, dy_track, ddy_track
    
    def roll_generator(self, goal=None, y0=None, tau=1.0, **kwargs):
        """
        Generator for real-time rollout of the DMP.

        Args:
            goal (np.array): Goal position of the DMP.
            y0 (np.array): Initial position of the DMP.
            tau (float): Temporal scaling factor.

        Yields:
            tuple: (position, velocity, acceleration) at each time step.
        """
        if goal is not None:
            self.goal = goal

        if y0 is not None:
            self.y0 = y0

        self.reset_state()  # Reset the system to initial conditions

        #timesteps = int(self.timesteps / tau)
        
        y = self.y0
        start_time_global = time.time()
        #elapsed_time_global = 0.0
        #for _ in range(timesteps):
        while y is not self.goal:
            # Compute the next step of the DMP
            
            if self.aruco_pose is None:
                rospy.logwarn("Aruco pose is not yet available, skipping step.")
                rospy.sleep(self.dt)
                continue

            # Use the latest pose from self.aruco_pose
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
            y, dy, ddy = self.step(tau=tau, pose = current_pose, **kwargs)
            # End timing
            end_time_step = time.time()

            # Calculate and print the time taken for this step
            computation_time = end_time_step - start_time_step
            #print(f"Computation time for this step: {computation_time:.6f} seconds")

            # Check if 40 seconds have passed since the loop started
            elapsed_time_global = time.time() - start_time_global
            yield y, dy, ddy



    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.y_old = self.y.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None, pose=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """
        #rospy.init_node("object_pose_subscriber")
        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):

            # Here some modifications could be done to
            # Account for coupling terms based on the distance...

            # generate the forcing term
            f = (self.gen_front_term(x, d) *
                 (np.dot(psi, self.w[d])) / np.sum(psi))
            # DMP acceleration
            self.ddy[d] = (self.ay[d] *
                           (self.by[d] * (self.goal[d] - self.y[d]) -
                           self.dy[d]/tau) + f) * tau
            if external_force is not None:
                self.ddy[d] += external_force[d]

            # Compute velocity and position
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            #rospy.loginfo(f"The pose of the aruco is: {self.aruco_pose}")
            euc_dist = np.sqrt(pose[0]**2 + pose[1]**2 + pose[2]**2)
            print(f"Euclidean distance: {euc_dist}")
            if euc_dist < 0.3:
                self.y[d] = self.y_old[d]
            else:
                self.y[d] += self.dy[d] * self.dt * error_coupling
            self.y_old[d] = self.y[d]

        return self.y, self.dy, self.ddy
