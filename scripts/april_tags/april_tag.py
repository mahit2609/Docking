import cv2
import apriltag
import numpy as np

class AprilTagPoseEstimator:
    def __init__(self, fx, fy, cx, cy, tag_size=0.2, R_extrinsic=None, T_extrinsic=None):
        """
        Initializes the AprilTagPoseEstimator class.

        :param fx, fy: Focal lengths in the x and y direction
        :param cx, cy: Optical center (principal point) coordinates in the image
        :param tag_size: Real-world size of the AprilTag (in meters, default 0.2m)
        :param R_extrinsic: Extrinsic rotation matrix (3x3) 
        :param T_extrinsic: Extrinsic translation vector (3x1) 
        """
        self.tag_size = tag_size  # The real-world size of the tag (in meters or any chosen unit)
        self.fx = fx  # Focal length in x
        self.fy = fy  # Focal length in y
        self.cx = cx  # Optical center in x
        self.cy = cy  # Optical center in y

        # If extrinsic parameters are provided, store them
        self.R_extrinsic = R_extrinsic if R_extrinsic is not None else np.eye(3)  # Identity matrix if not provided
        self.T_extrinsic = T_extrinsic if T_extrinsic is not None else np.zeros((3, 1))  # Zero translation if not provided

    def detect_apriltag(self, image):
        """
        Detects AprilTags in an image.
        
        :param image: The input image in which to detect AprilTags
        :return: List of detected tags (detections)
        """
        detector = apriltag.Detector()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray_image)
        return detections

    def estimate_pose(self, detection):
        """
        Estimates the pose of the detected AprilTag.
        Combines intrinsic and extrinsic parameters to get the pose.

        :param detection: A single detection from the AprilTag detector.
        :return: rotation vector (rvec), translation vector (tvec).
        """
        # 3D coordinates of the AprilTag corners in the tag coordinate frame (assuming the tag is at Z=0)
        tag_3d_points = np.array([
            [-self.tag_size / 2, -self.tag_size / 2, 0],
            [self.tag_size / 2, -self.tag_size / 2, 0],
            [self.tag_size / 2, self.tag_size / 2, 0],
            [-self.tag_size / 2, self.tag_size / 2, 0]
        ], dtype=np.float32)

        # The 4 corners of the detected AprilTag in the 2D image
        corners = detection.corners
        corners = corners.astype(np.float32)  # Ensure the corners are of type float32

        # Camera intrinsic matrix (3x3) built from class variables
        K = np.array([
            [self.fx, 0, self.cx],  # fx, 0, cx
            [0, self.fy, self.cy],  # 0, fy, cy
            [0, 0, 1]               # 0, 0, 1
        ], dtype=np.float32)

        # Use OpenCV's solvePnP to estimate pose (rotation and translation)
        success, rvec, tvec = cv2.solvePnP(tag_3d_points, corners, K, distCoeffs=None)

        if success:
            # Apply extrinsic parameters to adjust the pose estimation
            # Applying the extrinsic rotation (R_extrinsic) and translation (T_extrinsic)
            rvec = self.R_extrinsic.dot(rvec)
            tvec = self.R_extrinsic.dot(tvec) + self.T_extrinsic
            return rvec, tvec
        else:
            print("Pose estimation failed.")
            return None, None
