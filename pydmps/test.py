from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# LaTeX command
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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

def compute_quaternion_distance(q2,q1):
    #computes rotational angles, that represent distance between quaternions q2 and q1

    # Compute axis-angle representations of quat
    rot1 = qLog(q1) #this will be rotation angles
    rot2 = qLog(q2)
    # Compute the relative rotation: q2 * q1^-1
    relative_rot = rot2 * rot1.inv() 
    axis_angle = relative_rot.as_rotvec()
    return axis_angle

verbose_km = True
verbose_ks = True

# Human initial reference frame
rx_h0 = R.from_euler('x', 90, degrees=True)
ry_h0 = R.from_euler('y', -90, degrees=True)
rz_h0 = R.from_euler('z', 0, degrees=True)
R_h0 = rx_h0 * ry_h0 * rz_h0 # Absolute rotation of the fixed frame (as a rotation matrix)

# Human rotated hand (of 20 degrees)
'''
rx_h = R.from_euler('x', 90, degrees=True)
ry_h = R.from_euler('y', -130, degrees=True)
rz_h = R.from_euler('z', 0, degrees=True)
'''
rx_h = R.from_euler('x', 90, degrees=True)
ry_h = R.from_euler('y', -130, degrees=True)
rz_h = R.from_euler('z', 20, degrees=True)
R_h = rx_h * ry_h * rz_h 

# To be used for simulations: pass to quaternions
q_h0 = R_h0.as_quat()
q_h = R_h.as_quat()
print(f"The initial quaternion is: {q_h0}")
print(f"The final quaternion is: {q_h}")

sfghjfjghthg

# Compute the relative rotation matrix for the human
R_h_h0 = R_h0.inv() * R_h

# The rotation vectors is:
rot_vec_human = R_h_h0.as_rotvec()
print(f"The relative rotation vector is: {rot_vec_human}")
angle = np.linalg.norm(rot_vec_human)
print(f"The angle in radians is: {angle}, while in degrees is: {np.rad2deg(angle)}")
print(f"The adimensional rotation vector is: {rot_vec_human / angle}")

# Define the sigmoidla function
km = 2
a_dm = -10
delta_dm = 0.7
sigma_r = 1 + (km - 1) / (1 + np.exp(-a_dm * (angle - delta_dm)))
print(f"The values of sigma_r are: {sigma_r}")

# Compute the amplified rotation vector
amplified_rot_vec_human = sigma_r * rot_vec_human
print(f"The amplified rotation vector is: {amplified_rot_vec_human}")
angle_amplified = np.linalg.norm(amplified_rot_vec_human)
print(f"The angle in radians is: {angle_amplified}, while in degrees is: {np.rad2deg(angle_amplified)}")
print(f"The adimensional rotation vector is: {amplified_rot_vec_human / angle_amplified}")

# Back to absolute rotation matrix
R_h_amplified = R.from_rotvec(amplified_rot_vec_human)
R_h_B = R_h0 * R_h_amplified
print(f"The quaternion of the final rotation is: {R_h_B.as_quat()}")

if verbose_km:
    # print the sigmoidal function from 0 to pi
    angles = np.linspace(0, 1.5, 100)
    sigmas = 1 + (km - 1) / (1 + np.exp(-a_dm * (angles - delta_dm)))
    plt.figure(figsize=(10, 6))
    plt.plot(angles, sigmas)
    # Plot a red dot highlighting sigma_r
    plt.plot(angle, sigma_r, 'ro', label=rf'$\sigma_r(\theta_H^0)$: {sigma_r:.3f}')
    # Plot a vertial dashed line at 'angle', that stops on the sigmoidal curve
    plt.plot([angle, angle], [1, sigma_r], 'k--')
    # Plot an horizontal dashed line at sigma_r that stops on the sigmoidal curve
    plt.plot([0, angle], [sigma_r, sigma_r], 'k--')
    # Draw a dashed certical line at delta_dm (ranging from 1 to km)
    plt.plot([delta_dm, delta_dm], [1, km], 'b--', label=rf'$\delta_d$: {delta_dm}')


    plt.xlabel(r'$\theta$ (rad)', fontsize = 15)
    plt.ylabel('$\sigma_r$', fontsize = 15)
    #plt.title(r'\textbf{Sigmoidal function}', fontsize = 20)
    plt.grid(True)
    plt.legend(fontsize = 14)
    plt.xlim(0, 1.5)
    plt.ylim(1, km)
    plt.show()

if verbose_ks:
    # print the sigmoidal function from 0 to pi
    angles = np.linspace(0, 1.0, 100)
    ks = 0.7 # Vary this in between 0 and 1
    ay = 25.0
    a_d = -10
    delta_d = 0.174 # 10 degrees
    q_r = np.array([0.38261159, -0.5841248,  -0.55003034, 0.45811921])  # Reference quaternion (no rotation)
    distance_angles = compute_quaternion_distance(q_h, q_r) #distance in rad
    distance_orientation = np.linalg.norm(distance_angles)
    print(f"The distance orientation in radians is: {distance_orientation}")
    sigma_s = 1 / (1 + np.exp(a_d * (distance_orientation - delta_d)))
    sigmas = 1 / (1 + np.exp(a_d * (angles - delta_d)))
    ks_sigmas = ks * sigmas
    plt.figure(figsize=(10, 6))
    plt.plot(angles, sigmas)
    # Plot a red dot highlighting sigma_s
    plt.plot(distance_orientation, sigma_s, 'ro', label=rf'$\sigma_s(d(\bar{{\mathbf{{g_o}}}}, \mathbf{{q}}))$: {sigma_s:.3f}')
    # Plot a vertial dashed line at 'angle', that stops on the sigmoidal curve
    plt.plot([distance_orientation, distance_orientation], [0, sigma_s], 'k--')
    # Plot an horizontal dashed line at sigma_r that stops on the sigmoidal curve
    plt.plot([0, distance_orientation], [sigma_s, sigma_s], 'k--')
    # Draw a dashed certical line at delta_dm (ranging from 1 to km)
    plt.plot([delta_d, delta_d], [0, 1], 'b--', label=rf'$\delta_s$: {delta_d}')


    plt.xlabel(r'd($\bar{\mathbf{g_o}}, \mathbf{q}$)', fontsize = 20)
    plt.ylabel('$\sigma_s$', fontsize = 20)
    #plt.title(r'\textbf{Sigmoidal function}', fontsize = 20)
    plt.grid(True)
    plt.legend(fontsize = 14)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1)
    plt.show()



