import tinyik

# SCARA robot: 2 revolute joints, link lengths 15cm and 14cm
arm = tinyik.Actuator(['z', [15., 0., 0.], 'z', [14., 0., 0.]])

def inverse_kinematics(x, y):
    """
    Given target end-effector position (x, y),
    returns joint angles [theta1, theta2] in radians.
    """
    arm.ee = [x, y, 0.]
    return arm.angles.tolist()

def forward_kinematics(theta1, theta2):
    """
    Given joint angles in radians,
    returns end-effector position [x, y, z].
    """
    arm.angles = [theta1, theta2]
    return arm.ee.tolist()

if __name__ == "__main__":
    angles = inverse_kinematics(20., 10.)
    print(f"Joint angles for (20, 10): {angles}")
    pos = forward_kinematics(angles[0], angles[1])
    print(f"Forward kinematics check: {pos}")
