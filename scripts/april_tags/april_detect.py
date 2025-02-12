# main.py

import cv2
from april_tag import AprilTagPoseEstimator
import numpy as np

# Intrinsic parameters 
fx = 1000  # Focal length in x
fy = 1000  # Focal length in y
cx = 320   # Optical center in x 
cy = 240   # Optical center in y 

R_extrinsic = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]], dtype=np.float32)  # **calibrate
T_extrinsic = np.array([[1], [2], [3]], dtype=np.float32)  # **calibrate

apriltag_pose_estimator = AprilTagPoseEstimator(fx, fy, cx, cy, R_extrinsic=R_extrinsic, T_extrinsic=T_extrinsic)
cap = cv2.VideoCapture('/dev/video5') #****

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    detections = apriltag_pose_estimator.detect_apriltag(frame)

    for detection in detections:
        rvec, tvec = apriltag_pose_estimator.estimate_pose(detection)

        if rvec is not None and tvec is not None:
            print(f"Rotation Vector (rvec): {rvec}")
            print(f"Translation Vector (tvec): {tvec}")

            corners = detection.corners.astype(int)
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]]).reshape(-1, 3)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), distCoeffs=None)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (255, 0, 0), 5)  # X-axis in blue
            cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 5)  # Y-axis in green
            cv2.line(frame, tuple(imgpts[1]), tuple(imgpts[2]), (0, 0, 255), 5)  # Z-axis in red

    cv2.imshow('AprilTag Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
