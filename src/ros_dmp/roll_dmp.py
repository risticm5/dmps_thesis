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
        print(f"Loaded weights: {weights}")
        n_dmps = weights.shape[0]
        n_bfs = weights.shape[1]
        self.dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=n_dmps, n_bfs=n_bfs,
                                                     dt=dt, ay=None, w=weights)

    def roll(self, goal, initial_pos, tau):
        ''' Generates trajectory for given parameters

        goal: Numpy array, goal vector
        initial_pos: Numpy array, initial pose vector
        tau: Time scaling factor
        '''

        # 'rollout' gives you the complete trajectory vectors
        self.pos, self.vel, self.acc = self.dmp.rollout(goal=goal, y0=initial_pos, tau=tau)
        return self.pos, self.vel, self.acc
    
    def roll_generator(self, goal, initial_pos, tau):
        """
        Generates trajectory dynamically for given parameters.

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
        
        with open(file_name) as f:
            loadeddict = yaml.safe_load(f)
        x = loadeddict.get('x')
        y = loadeddict.get('y')
        z = loadeddict.get('z')
        qx = loadeddict.get('qx')
        qy = loadeddict.get('qy')
        qz = loadeddict.get('qz')
        qw = loadeddict.get('qw')

        weights = np.array(x)
        weights = np.vstack((weights, np.array(y)))
        weights = np.vstack((weights, np.array(z)))
        weights = np.vstack((weights, np.array(qx)))
        weights = np.vstack((weights, np.array(qy)))
        weights = np.vstack((weights, np.array(qz)))
        weights = np.vstack((weights, np.array(qw)))

        return weights

