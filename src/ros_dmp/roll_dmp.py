import numpy as np
import yaml
import os.path
import pydmps
import rospy

class RollDmp():

    def __init__(self, file_name, dt):
        ''' Interface for Generating path from learned DMP weights
        
        file_name: Path of the weight file
        dt: Time step size for integration of differential equations
        '''
        self.pos = None
        self.vel = None
        self.acc = None
        weights = self.load_weights(file_name)
        n_dmps = weights.shape[0]
        n_bfs = weights.shape[1]
        self.dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=n_dmps, n_bfs=n_bfs,
                                                     dt=dt, ay=None, w=weights)

    def roll(self, goal, initial_pos, tau,**kwargs):
        ''' Generates trajectory for given parameters

        goal: Numpy array, goal vector
        initial_pos: Numpy array, initial pose vector
        tau: Time scaling factor 
        '''
        #Normal execution - without GLISp optimiz of ay and by
        self.pos, self.vel, self.acc = self.dmp.rollout(goal=goal, y0=initial_pos, tau=tau)
        #GLISp execution - wit GLISp optimiz of ay and by
        if 'ay' in kwargs and 'by' in kwargs:
            ay_glisp=kwargs['ay']
            by_glisp=kwargs['by']
            self.pos, self.vel, self.acc = self.dmp.rollout(goal=goal, y0=initial_pos, tau=tau, ay_glisp=ay_glisp, by_glisp=by_glisp)
        return self.pos, self.vel, self.acc

    def roll_generator(self, goal, initial_pos, tau):
        """
        Generates trajectory dynamically for given parameters.
        Christian added this function, to execute dmps real_time

        Args:
            goal (np.array): Goal position of the DMP.
            initial_pos (np.array): Initial position of the DMP.
            tau (float): Temporal scaling factor.

        Yields:
            tuple: (position, velocity, acceleration) at each time step.
        """
        yield from self.dmp.roll_generator(goal=goal, y0=initial_pos, tau=tau)

    def load_weights(self, file_name):
        '''Loads DMP weights from given weight file

        file_name: Path of the weight file
        '''
        #for Euler DMPs
        """
        with open(file_name) as f:
            loadeddict = yaml.safe_load(f)
        x = loadeddict.get('x')
        y = loadeddict.get('y')
        z = loadeddict.get('z')
        roll = loadeddict.get('roll')
        pitch = loadeddict.get('pitch')
        yaw = loadeddict.get('yaw')

        weights = np.array(x)
        weights = np.vstack((weights, np.array(y)))
        weights = np.vstack((weights, np.array(z)))
        weights = np.vstack((weights, np.array(roll)))
        weights = np.vstack((weights, np.array(pitch)))
        weights = np.vstack((weights, np.array(yaw)))
        """

        #for Quaternion DMPs
        with open(file_name) as f:
            loadeddict = yaml.safe_load(f)
        x = loadeddict.get('x')
        y = loadeddict.get('y')
        z = loadeddict.get('z')
        rx = loadeddict.get('rx')
        ry = loadeddict.get('ry')
        rz = loadeddict.get('rz')

        weights = np.array(x)
        weights = np.vstack((weights, np.array(y)))
        weights = np.vstack((weights, np.array(z)))
        weights = np.vstack((weights, np.array(rx)))
        weights = np.vstack((weights, np.array(ry)))
        weights = np.vstack((weights, np.array(rz)))

        return weights

