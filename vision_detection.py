"""
vision_detection.py
──────────────────────────────────────────────────────────────────────────────
Real-time object detection and classification for the SCARA vision inspection
system using OpenCV. Detects objects on a conveyor/workspace by color and
shape, classifies them (GOOD / DEFECTIVE), and returns their real-world
coordinates for the robot arm to pick.
──────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Camera & Workspace Calibration ───────────────────────────────────────────
CAMERA_INDEX    = 0         # USB/Pi camera index
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
WORKSPACE_WIDTH_CM  = 30   # Real workspace width  (cm)
WORKSPACE_HEIGHT_CM = 25   # Real workspace height (cm)

# Pixels per cm (calibrate with known reference object)
PX_PER_CM_X = FRAME_WIDTH  / WORKSPACE_WIDTH_CM
PX_PER_CM_Y = FRAME_HEIGHT / WORKSPACE_HEIGHT_CM

# ── HSV Color Ranges for Classification ──────────────────────────────────────
COLOR_RANGES = {
    "RED":    ([0,   120,  70], [10,  255, 255]),
    "GREEN":  ([36,   50,  50], [86,  255, 255]),
    "BLUE":   ([100,  50,  50], [130, 255, 255]),
    "YELLOW": ([20,  100, 100], [30,  255, 255]),
}

MIN_CONTOUR_AREA = 500   # pixels² — ignore noise

# ── Detected Object ───────────────────────────────────────────────────────────
@dataclass
class DetectedObject:
    color:      str
    shape:      str
    cx_px:      int     # centroid X in pixels
    cy_px:      int     # centroid Y in pixels
    cx_cm:      float   # centroid X in cm (robot coordinates)
    cy_cm:      float   # centroid Y in cm (robot coordinates)
    area:       float
    status:     str     # "GOOD" or "DEFECTIVE"

# ── Pixel → Real-World Coordinate ────────────────────────────────────────────
def pixel_to_cm(px: int, py: int):
    cx_cm = px / PX_PER_CM_X
    cy_cm = (FRAME_HEIGHT - py) / PX_PER_CM_Y   # flip Y (camera vs robot frame)
    return round(cx_cm, 2), round(cy_cm, 2)

# ── Shape Classification ──────────────────────────────────────────────────────
def classify_shape(contour) -> str:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:   return "TRIANGLE"
    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h)
        return "SQUARE" if 0.85 <= aspect <= 1.15 else "RECTANGLE"
    if vertices >= 8:   return "CIRCLE"
    return "POLYGON"

# ── Color Detection ───────────────────────────────────────────────────────────
def detect_color(roi_hsv) -> str:
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 50:
            return color_name
    return "UNKNOWN"

# ── Quality Check (example: defective if area too small or color unknown) ─────
def check_quality(obj: DetectedObject) -> str:
    if obj.color == "UNKNOWN":      return "DEFECTIVE"
    if obj.area  < MIN_CONTOUR_AREA * 1.5: return "DEFECTIVE"
    return "GOOD"

# ── Process One Frame ─────────────────────────────────────────────────────────
def process_frame(frame) -> list:
    detected = []

    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # ROI around centroid for color detection
        r = 15
        roi = hsv[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
        color  = detect_color(roi) if roi.size > 0 else "UNKNOWN"
        shape  = classify_shape(cnt)
        cx_cm, cy_cm = pixel_to_cm(cx, cy)

        obj = DetectedObject(
            color=color, shape=shape,
            cx_px=cx, cy_px=cy,
            cx_cm=cx_cm, cy_cm=cy_cm,
            area=area, status=""
        )
        obj.status = check_quality(obj)
        detected.append(obj)

        # Draw on frame
        color_bgr = (0, 255, 0) if obj.status == "GOOD" else (0, 0, 255)
        cv2.drawContours(frame, [cnt], -1, color_bgr, 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
        cv2.putText(frame, f"{obj.color} {obj.shape}", (cx-40, cy-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
        cv2.putText(frame, f"{obj.status}", (cx-30, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)
        cv2.putText(frame, f"({cx_cm:.1f},{cy_cm:.1f})cm", (cx-40, cy+38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1)

    return detected, frame

# ── Main Loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Vision Detection Running — press 'q' to quit")
    print(f"Workspace: {WORKSPACE_WIDTH_CM}cm × {WORKSPACE_HEIGHT_CM}cm")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        objects, annotated = process_frame(frame)

        for obj in objects:
            status_icon = "✅" if obj.status == "GOOD" else "❌"
            print(f"  {status_icon} {obj.status:9s} | {obj.color:7s} {obj.shape:10s} | "
                  f"Pixel:({obj.cx_px:3d},{obj.cy_px:3d}) | "
                  f"Real:({obj.cx_cm:.1f},{obj.cy_cm:.1f})cm | Area:{obj.area:.0f}")

        cv2.putText(annotated, f"Objects: {len(objects)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("SCARA Vision Inspection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
