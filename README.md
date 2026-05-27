# SCARA Robot for Vision Inspection Application

> Project — Department of Robotics and Automation  
> Sri Ramakrishna Engineering College, Coimbatore | Anna University, 2022–2023

---

## Overview

A 3-axis **SCARA robot arm** integrated with a **TensorFlow 2 object detection pipeline** for automated industrial quality inspection. The system combines robot kinematics solved in Python, real-time SSD MobileNet V2 inference on Raspberry Pi 4, and servo actuation via PCA9685 to perform two-mode inspection: continuous conveyor-based outer-surface checking and batchwise internal hole inspection.

---

## System Architecture

```
[Pi Camera]
     |
     v
[TensorFlow 2 - SSD MobileNet V2]
 |-- Trained on custom dataset (LabelImg annotations)
 |-- GPU-trained, exported as frozen .pb file
 |-- Deployed on Raspberry Pi 4 for real-time inference
     |
     v
[Vision Output - XY coordinates of detected object]
     |
     v
[Kinematics Engine - Tinyik]
 |-- Inverse kinematics -> joint angles (theta1, theta2)
 |-- Forward kinematics -> end-effector position
     |
     v
[Servo Controller - PCA9685 over I2C]
 |-- Joint 3 - Arm 1 rotation (revolute, Z-axis, +/-360 degrees)
 |-- Joint 2 - Arm 2 rotation (revolute, Z-axis, -30 to +270 degrees)
 |-- Joint 1 - Z-axis linear motion (0 to 10cm)
```

## Hardware

| Component | Specification |
|---|---|
| SCARA Robot | 3-axis, 750g, 29cm reach, 12.56 rad/s joint velocity, 110g payload |
| Servo Motors | MG90 micro servos |
| Servo Driver | PCA9685 16-channel PWM driver (I2C) |
| Controller | Raspberry Pi 4 |
| Camera | Pi Camera module (CSI) |

---

## Inspection Pipeline

Two-mode inspection system:

**Continuous checking** — Pi Camera streams live feed over the conveyor belt. TensorFlow model detects product presence. RGB pixel boundary analysis checks the outer surface. A quick-return mechanism ejects defective products automatically without manual intervention.

**Batchwise checking** — Vision system detects XY coordinates of each product. Tinyik solves inverse kinematics to compute joint angles. SCARA arm physically picks and repositions the product for internal hole diameter inspection.

---

## ML Model

- Model: TensorFlow 2 SSD MobileNet V2 (Single Shot Detector)
- Chosen for low-latency inference on constrained hardware (Raspberry Pi 4)
- Dataset: custom images of sharpeners annotated with LabelImg, split 90% train / 10% test
- Training: GPU-trained, loss monitored via TensorBoard, stopped when total loss < 0.05
- Deployment: exported as frozen .pb file, loaded on Raspberry Pi 4 at runtime

---

## Kinematics

Inverse and forward kinematics solved using the **Tinyik** Python library.

- Link lengths: Arm 1 = 15cm, Arm 2 = 14cm
- Input: target XY coordinates from vision pipeline
- Output: joint angles θ1, θ2 sent as PWM signals to servos via PCA9685

```python
import tinyik
arm = tinyik.Actuator(['z', [15., 0., 0.], 'z', [14., 0., 0.]])
arm.ee = [x, y, 0.]   # set target position
angles = arm.angles    # get joint angles
```

---

## Repository Files

| File | Description |
|---|---|
| `scara_kinematics.py` | Forward and inverse kinematics using Tinyik — takes target XY, returns joint angles |
| `vision_detection.py` | TensorFlow 2 SSD MobileNet V2 inference — loads .pb model, runs detection on Pi Camera feed |

---

## Tools & Technologies

`Python` `TensorFlow 2` `SSD MobileNet V2` `LabelImg` `Tinyik` `OpenCV` `Raspberry Pi 4` `PCA9685` `MG90 Servos` `Pi Camera`
