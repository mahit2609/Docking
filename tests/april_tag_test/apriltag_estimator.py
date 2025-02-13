import cv2
import apriltag
import numpy as np

class AprilTagPoseEstimator:
    def __init__(self, fx, fy, cx, cy, tag_size=0.2, R_extrinsic=None, T_extrinsic=None, 
                 distortion_coeffs=None, x_scale=1.0, y_scale=1.0, z_scale=1.0):
        """
        Initializes the AprilTagPoseEstimator class.
        :param fx, fy: Focal lengths in the x and y direction
        :param cx, cy: Optical center (principal point) coordinates in the image
        :param tag_size: Real-world size of the AprilTag (in meters, default 0.2m)
        :param R_extrinsic: Extrinsic rotation matrix (3x3)
        :param T_extrinsic: Extrinsic translation vector (3x1)
        :param distortion_coeffs: Camera distortion coefficients [k1, k2, p1, p2, k3]
        :param x_scale, y_scale, z_scale: Scaling factors for each axis
        """
        self.tag_size = tag_size
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # If extrinsic parameters are provided, store them
        self.R_extrinsic = R_extrinsic if R_extrinsic is not None else np.eye(3)
        self.T_extrinsic = T_extrinsic if T_extrinsic is not None else np.zeros((3, 1))
        
        # Add distortion coefficients
        self.dist_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5)
        
        # Add scaling factors
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale

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
        corners = corners.astype(np.float32)

        # Camera intrinsic matrix (3x3) built from class variables
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Use OpenCV's solvePnP to estimate pose (rotation and translation)
        success, rvec, tvec = cv2.solvePnP(tag_3d_points, corners, K, self.dist_coeffs)

        if success:
            # Apply extrinsic parameters to adjust the pose estimation
            # Applying the extrinsic rotation (R_extrinsic) and translation (T_extrinsic)
            rvec = self.R_extrinsic.dot(rvec)
            tvec = self.R_extrinsic.dot(tvec) + self.T_extrinsic
            
            # Apply scaling
            tvec[0] *= self.x_scale
            tvec[1] *= self.y_scale
            tvec[2] *= self.z_scale
            
            return rvec, tvec
        else:
            print("Pose estimation failed.")
            return None, None

    def draw_tag(self, image, detection, pose=None):
        """
        Draws the detected AprilTag on the image with ID and axes if pose is available.
        :param image: Input image
        :param detection: AprilTag detection
        :param pose: (Optional) Tuple of (rvec, tvec) from pose estimation
        :return: Image with visualization
        """
        # Draw tag outline
        corners = detection.corners.astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        
        # Draw tag center
        center = tuple(map(int, detection.center))
        cv2.circle(image, center, 4, (0, 0, 255), -1)
        
        # Draw tag ID
        cv2.putText(image, f"ID: {detection.tag_id}", 
                   (corners[0][0], corners[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # If pose is available, draw axes
        if pose is not None and pose[0] is not None:
            rvec, tvec = pose
            K = np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)
            cv2.drawFrameAxes(image, K, self.dist_coeffs, rvec, tvec, self.tag_size/2)
        
        return image