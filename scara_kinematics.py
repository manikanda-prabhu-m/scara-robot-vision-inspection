"""
scara_kinematics.py
──────────────────────────────────────────────────────────────────────────────
Forward and Inverse Kinematics for a 2-DOF planar SCARA robot arm.

SCARA Joint Configuration:
  Joint 1 (θ1) — Shoulder: rotates about Z-axis
  Joint 2 (θ2) — Elbow:    rotates about Z-axis
  Link lengths: L1 (upper arm), L2 (forearm)

Forward Kinematics:
  Given θ1, θ2 → compute end-effector (x, y)

Inverse Kinematics:
  Given target (x, y) → compute θ1, θ2
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Robot Parameters ──────────────────────────────────────────────────────────
L1 = 15.0   # Upper arm length (cm)
L2 = 12.0   # Forearm length  (cm)

THETA1_MIN, THETA1_MAX = -150, 150   # degrees
THETA2_MIN, THETA2_MAX = -120, 120   # degrees

# ── Forward Kinematics ────────────────────────────────────────────────────────
def forward_kinematics(theta1_deg: float, theta2_deg: float):
    """
    Compute end-effector position (x, y) given joint angles.

    Args:
        theta1_deg: Shoulder joint angle in degrees
        theta2_deg: Elbow joint angle in degrees

    Returns:
        (x, y): End-effector Cartesian coordinates (cm)
        (elbow_x, elbow_y): Elbow position for visualization
    """
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)

    elbow_x = L1 * np.cos(theta1)
    elbow_y = L1 * np.sin(theta1)

    x = elbow_x + L2 * np.cos(theta1 + theta2)
    y = elbow_y + L2 * np.sin(theta1 + theta2)

    return (round(x, 3), round(y, 3)), (round(elbow_x, 3), round(elbow_y, 3))

# ── Inverse Kinematics ────────────────────────────────────────────────────────
def inverse_kinematics(x: float, y: float, elbow_up: bool = True):
    """
    Compute joint angles to reach target (x, y).

    Args:
        x, y    : Target end-effector position (cm)
        elbow_up: True = elbow-up configuration, False = elbow-down

    Returns:
        (theta1_deg, theta2_deg) or None if target unreachable
    """
    d = np.sqrt(x**2 + y**2)

    # Check reachability
    if d > (L1 + L2):
        print(f"  [IK] Target ({x}, {y}) is OUT OF REACH. Max reach = {L1+L2:.1f} cm")
        return None
    if d < abs(L1 - L2):
        print(f"  [IK] Target ({x}, {y}) is TOO CLOSE. Min reach = {abs(L1-L2):.1f} cm")
        return None

    # Law of cosines for theta2
    cos_theta2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    if not elbow_up:
        sin_theta2 = -sin_theta2

    theta2 = np.arctan2(sin_theta2, cos_theta2)

    # Theta1
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)

    # Check joint limits
    if not (THETA1_MIN <= theta1_deg <= THETA1_MAX):
        print(f"  [IK] θ1={theta1_deg:.1f}° exceeds joint limit [{THETA1_MIN}, {THETA1_MAX}]")
        return None
    if not (THETA2_MIN <= theta2_deg <= THETA2_MAX):
        print(f"  [IK] θ2={theta2_deg:.1f}° exceeds joint limit [{THETA2_MIN}, {THETA2_MAX}]")
        return None

    return round(theta1_deg, 2), round(theta2_deg, 2)

# ── Workspace Boundary Check ──────────────────────────────────────────────────
def is_reachable(x: float, y: float) -> bool:
    d = np.sqrt(x**2 + y**2)
    return abs(L1 - L2) <= d <= (L1 + L2)

# ── Visualize Robot Arm ───────────────────────────────────────────────────────
def visualize(theta1_deg: float, theta2_deg: float, target=None):
    (x_ee, y_ee), (x_elbow, y_elbow) = forward_kinematics(theta1_deg, theta2_deg)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-(L1+L2+2), (L1+L2+2))
    ax.set_ylim(-(L1+L2+2), (L1+L2+2))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    # Workspace circles
    outer = plt.Circle((0,0), L1+L2, fill=False, color="lightblue", lw=1, linestyle="--", label=f"Max reach ({L1+L2:.0f}cm)")
    inner = plt.Circle((0,0), abs(L1-L2), fill=False, color="lightyellow", lw=1, linestyle="--", label=f"Min reach ({abs(L1-L2):.0f}cm)")
    ax.add_patch(outer)
    ax.add_patch(inner)

    # Robot links
    ax.plot([0, x_elbow], [0, y_elbow], "b-", lw=4, label=f"Link 1 (L={L1}cm)")
    ax.plot([x_elbow, x_ee], [y_elbow, y_ee], "r-", lw=3, label=f"Link 2 (L={L2}cm)")

    # Joints
    ax.plot(0, 0, "ko", markersize=10, zorder=5, label="Base (origin)")
    ax.plot(x_elbow, y_elbow, "bo", markersize=8, zorder=5, label="Elbow joint")
    ax.plot(x_ee, y_ee, "r*", markersize=14, zorder=5, label="End-effector")

    if target:
        ax.plot(target[0], target[1], "gx", markersize=12, mew=2, label="Target")

    ax.set_title(f"SCARA Robot — θ1={theta1_deg:.1f}°  θ2={theta2_deg:.1f}°\n"
                 f"End-effector: ({x_ee:.2f}, {y_ee:.2f}) cm", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("scara_visualization.png", dpi=150)
    plt.show()
    print("Visualization saved as scara_visualization.png")

# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SCARA ROBOT KINEMATICS")
    print(f"  Link lengths: L1={L1}cm, L2={L2}cm")
    print("=" * 55)

    # Forward kinematics demo
    print("\n── Forward Kinematics ──────────────────────────────")
    test_angles = [(45, 30), (90, -45), (30, 60), (0, 90)]
    for t1, t2 in test_angles:
        (x, y), (ex, ey) = forward_kinematics(t1, t2)
        print(f"  θ1={t1:5.1f}°  θ2={t2:5.1f}°  →  EE: ({x:6.2f}, {y:6.2f}) cm  |  Elbow: ({ex:.2f}, {ey:.2f})")

    # Inverse kinematics demo
    print("\n── Inverse Kinematics ──────────────────────────────")
    targets = [(20, 10), (15, 15), (5, 5), (0, 25), (30, 0)]
    for tx, ty in targets:
        result = inverse_kinematics(tx, ty)
        if result:
            t1, t2 = result
            (vx, vy), _ = forward_kinematics(t1, t2)
            print(f"  Target: ({tx:5.1f}, {ty:5.1f})  →  θ1={t1:7.2f}°  θ2={t2:7.2f}°  |  Verify: ({vx:.2f}, {vy:.2f})")
        else:
            print(f"  Target: ({tx:5.1f}, {ty:5.1f})  →  Unreachable")

    # Visualize one configuration
    print("\n── Visualizing θ1=45°, θ2=30° ─────────────────────")
    visualize(45, 30, target=(20, 10))
