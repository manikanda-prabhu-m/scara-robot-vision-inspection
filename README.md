# SCARA Robot for Vision Inspection Application

> Project — Department of Robotics and Automation  
> Sri Ramakrishna Engineering College, Coimbatore | Anna University, 2022–2023

---

## Overview

A **SCARA (Selective Compliance Articulated Robot Arm)** integrated with a **camera-based vision inspection system** for automated object detection, classification, and pick-and-place operations. The system combines robot kinematics, servo control, and OpenCV computer vision to enable intelligent, autonomous industrial inspection.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCARA ROBOT SYSTEM                        │
│                                                             │
│   [Pi Camera / USB Camera]                                  │
│          │                                                  │
│          ▼                                                  │
│   [OpenCV Vision Module]                                    │
│    ├── Object Detection (contour / color / shape)           │
│    ├── Classification (good / defective / type)             │
│    └── Position Calculation (pixel → real-world coords)     │
│          │                                                  │
│          ▼                                                  │
│   [Kinematics Engine]                                       │
│    ├── Inverse Kinematics → joint angles (θ1, θ2)          │
│    └── Workspace boundary checking                          │
│          │                                                  │
│          ▼                                                  │
│   [Servo Controller]                                        │
│    ├── Joint 1 — Shoulder rotation (servo 1)               │
│    ├── Joint 2 — Elbow rotation    (servo 2)               │
│    ├── Joint 3 — Z-axis vertical   (servo 3)               │
│    └── Joint 4 — End-effector grip (servo 4)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Components

| Component | Purpose |
|-----------|---------|
| SCARA Robot Arm (4-DOF) | Pick-and-place manipulation |
| Servo Motors (×4) | Joint actuation (shoulder, elbow, Z, gripper) |
| PCA9685 PWM Driver | Multi-servo control via I²C |
| Raspberry Pi 3 / USB Camera | Vision capture and processing |
| OpenCV | Real-time image processing |
| Python | Control and vision software |

---

## Key Features

- **Real-time object detection** using OpenCV contour analysis and color segmentation
- **Inverse kinematics** to compute joint angles from target (x, y) coordinates
- **Automated classification** — separates objects by color, shape, or size
- **Pick-and-place** — robot arm moves to detected object, picks it, places in bin
- **Workspace safety** — boundary checking prevents arm from exceeding joint limits

---

## Files

| File | Description |
|------|-------------|
| `scara_kinematics.py` | Forward and inverse kinematics for 2-DOF planar SCARA |
| `vision_detection.py` | OpenCV object detection, classification, coordinate extraction |
| `servo_controller.py` | PCA9685 servo control — maps joint angles to PWM signals |
| `main_inspection.py` | Main integration — vision → kinematics → servo → pick-and-place |

---

## Tools & Technologies

- Python 3, OpenCV, NumPy
- Raspberry Pi 3, PCA9685, Servo motors
- Robot kinematics (forward + inverse)
- Computer vision (contour detection, color segmentation, HSV thresholding)
