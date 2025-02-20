from scipy.spatial.transform import Rotation as R
import numpy as np

rx = R.from_euler('x', 90, degrees=True)
ry = R.from_euler('y', -70, degrees=True)
rz = R.from_euler('z', 0, degrees=True)
fixed_ref_matrices = rx * ry * rz # Absolute rotation of the fixed frame (as a rotation matrix)
fixed_ref_eul_ang = fixed_ref_matrices.as_euler('xyz', degrees=True)
print(f"The fixed reference frame in euler angles is: {fixed_ref_eul_ang}")

# Compute the quaternion
fixed_ref_quat = fixed_ref_matrices.as_quat()
print(f"The fixed reference frame in quaternion is: {fixed_ref_quat}")

angle = 50 * np.pi / 180
print(f"The angle in radians is: {angle}")

quat_dmp_link = np.array([0.20769, -0.31022, -0.67978, 0.63127])
eul_dmp_link = R.from_quat(quat_dmp_link).as_euler('xyz', degrees=True)
print(f"The DMP link frame in euler angles is: {eul_dmp_link}")