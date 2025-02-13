import cv2
import numpy as np
from apriltag_estimator import AprilTagPoseEstimator  # Import your class from the other file

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Camera parameters (replace with your calibrated values)
    fx = 1.74266348e+03
    fy = 1.73815941e+03
    cx = 8.55750077e+02
    cy = 5.54921581e+02
    
    # Distortion coefficients
    dist_coeffs = np.array([-0.50980866, 1.00789444, -0.00915618, 0.00429768, -1.2860852])
    
    # Initialize pose estimator
    estimator = AprilTagPoseEstimator(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        tag_size=0.02,  # 2cm tag size
        distortion_coeffs=dist_coeffs,
        x_scale=1.765,
        y_scale=1.765,
        z_scale=1.582
    )
    
    # Initialize video writer
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect AprilTags
            detections = estimator.detect_apriltag(frame)
            
            if len(detections) > 0:
                for detection in detections:
                    # Get pose
                    pose = estimator.estimate_pose(detection)
                    
                    if pose[0] is not None:  # If pose estimation succeeded
                        # Draw visualization
                        frame = estimator.draw_tag(frame, detection, pose)
                        
                        # Extract and print position
                        rvec, tvec = pose
                        x, y, z = tvec.flatten()
                        print(f"Tag {detection.tag_id} position: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
            else:
                print("No tags detected")
            
            # Write frame
            #out.write(frame)
            
            # Display result
            cv2.imshow('AprilTag Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        #out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()