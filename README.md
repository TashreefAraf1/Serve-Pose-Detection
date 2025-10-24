# 🎾 Tennis Serve Pose and Motion Analysis using MediaPipe + YOLO

This project performs **detailed analysis of tennis serves** using **pose estimation (MediaPipe)** and **object detection (YOLO)**.  
It extracts **kinematic features** such as wrist rotation, arm speed, angular velocity, body tilt, and service motion timing — enabling quantitative evaluation of a player’s serve mechanics.

---

## 🧠 Overview

The pipeline processes tennis serve videos frame-by-frame and performs:

1. **Human pose estimation** using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose).
2. **Object detection** (ball and racquet) using [YOLOv8](https://github.com/ultralytics/ultralytics).
3. **Motion analysis** to compute biomechanical metrics such as:
   - Wrist angular velocity and rotation
   - Arm and knee flexion angles
   - Body tilt and trunk angle
   - Wrist arc curvature (trajectory smoothness)
   - Service motion timing (ball release → racquet contact)
4. **Data logging** to CSV and **video output** with overlaid pose landmarks and motion data.



