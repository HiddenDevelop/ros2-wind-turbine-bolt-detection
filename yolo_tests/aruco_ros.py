from matplotlib.pyplot import gray

import rclpy
import cv2 as cv
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from ultralytics import YOLO

class YoloDepthLocalization(Node):
    def __init__(self):
        super().__init__('yolo_depth_localization')

        self.bridge = CvBridge()
        
        # Load your custom trained YOLO model
        # Replace 'best.pt' with the path to your trained bolt model
        self.model = YOLO('best.pt') 

        # Camera Intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.T_M_C = None #Transformation from Camera to Mockup frame
        self.T_M_A = np.eye(4, dtype=np.float64)

        bolt_points_cm = np.array([
            [ -9.5,  91.0, 0.0],
            [ -5.0,  83.0, 0.0],
            [-14.0,  82.0, 0.0],
            [-19.0,  90.0, 0.0],
            [-28.0,  88.0, 0.0],
            [-24.0,  79.0, 0.0],
            [-37.5,  84.0, 0.0],
            [-32.5,  76.0, 0.0],
            [-46.0,  80.0, 0.0],
            [-40.0,  72.0, 0.0],
            [-53.0,  75.0, 0.0],
            [-49.0,  67.0, 0.0],
            [-61.0,  71.0, 0.0],
            [-57.0,  60.0, 0.0],
            [-63.0,  53.0, 0.0],
            [-75.0,  54.0, 0.0],
            [-69.0,  45.0, 0.0],
            [-74.0,  37.0, 0.0],
            [-84.5,  38.0, 0.0],
            [-78.0,  29.0, 0.0],
            [-88.0,  30.0, 0.0],
            [-82.0,   8.0, 0.0],
            [-90.5,   8.0, 0.0],
            [-57.5, 101.0, 0.0],
            [-101.0, 58.0, 0.0]
        ], dtype=np.float64)

        self.bolt_points_M = (bolt_points_cm * 0.01)

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        # Synchronize RGB and depth frames
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.kinect_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/k4a/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.target_pub = self.create_publisher(Point, '/hex_bolt/spatial_coordinate', 10)

        self.get_logger().info('YOLO 3D Target Tracker started. Waiting for frames...')


    def camera_info_callback(self, msg):
        # Extract intrinsic parameters once
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(f'Intrinsics locked: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

        self.camera_matrix = np.array(
            msg.k,
            dtype=np.float64
        ).reshape(3, 3)

        if len(msg.d) > 0:
            self.dist_coeffs = np.array(
                msg.d,
                dtype=np.float64
            ).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)


    def aruco_display(corners, ids, rejected, image):
        
        if len(corners) > 0:
            
            ids = ids.flatten()
            
            for (markerCorner, markerID) in zip(corners, ids):
                
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                
                cv.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID: {}".format(markerID))
                
        return image

    def pose_estimation(self,frame,detector,matrix_coefficients,distortion_coefficients,MARKER_SIZE_M):

        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

        corners, ids, rejected_img_points = detector.detectMarkers(
            gray
        )

        if ids is not None:

            cv.aruco.drawDetectedMarkers(
                frame,
                corners,
                ids
            )

            ids_flat = ids.flatten()

            for i, marker_id in enumerate(ids_flat):

                print(
                    f"Detected ArUco ID: {int(marker_id)}"
                )

                # ---------------------------------------------
                # Known physical marker geometry
                # ---------------------------------------------

                half = MARKER_SIZE_M / 2.0

                marker_points_3d = np.array([
                    [-half,  half, 0.0],   # top-left
                    [ half,  half, 0.0],   # top-right
                    [ half, -half, 0.0],   # bottom-right
                    [-half, -half, 0.0]    # bottom-left
                ], dtype=np.float32)

                # Detected image pixels for this marker
                image_points = corners[i].reshape(
                    4,
                    2
                ).astype(np.float32)

                # ---------------------------------------------
                # PnP pose estimation
                # ---------------------------------------------

                success, rvec, tvec = cv.solvePnP(
                    marker_points_3d,
                    image_points,
                    matrix_coefficients,
                    distortion_coefficients,
                    flags=cv.SOLVEPNP_IPPE_SQUARE
                )

                if success:                    
                    # Translation of marker origin
                    # expressed in CAMERA coordinates
                    X = float(tvec[0, 0])
                    Y = float(tvec[1, 0])
                    Z = float(tvec[2, 0])

                    print(
                        f"Marker position in camera frame: "
                        f"X={X:.3f}, Y={Y:.3f}, Z={Z:.3f} m"
                    )
                if not success:
                    continue
                # Draw marker coordinate frame
                cv.drawFrameAxes(
                    frame,
                    matrix_coefficients,
                    distortion_coefficients,
                    rvec,
                    tvec,
                    MARKER_SIZE_M * 0.75
                )
                
                R_C_A, _ = cv.Rodrigues(rvec)

                T_C_A = np.eye(4, dtype=np.float64)

                T_C_A[:3, :3] = R_C_A
                T_C_A[:3, 3] = tvec.reshape(3)

                return frame, T_C_A

        return frame, None

    def associate_bolt(self,point_M,max_distance_m=0.10):

        # We currently care about the position
        # projected onto the mockup XY plane.
        measured_xy = point_M[:2]

        known_xy = self.bolt_points_M[:, :2]

        # Residual for every known bolt
        residuals = (known_xy -measured_xy)

        # Least-squares cost:
        #
        # J = dx^2 + dy^2
        squared_errors = np.sum(residuals ** 2,axis=1)

        best_index = int(np.argmin(squared_errors))

        distance_error = float(np.sqrt(squared_errors[best_index]))

        if distance_error > max_distance_m:
            return None, None, distance_error

        matched_position = (self.bolt_points_M[best_index])

        return (best_index,matched_position,distance_error)

    def kinect_callback(self, rgb_msg, depth_msg):
        if self.fx is None:
            return

        MARKER_SIZE_M = 0.1778

        ARUCO_DICT = {
            "DICT_4X4_50": cv.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
        }

        aruco_type = "DICT_5X5_100"

        arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

        arucoParams = cv.aruco.DetectorParameters()
        Detector = cv.aruco.ArucoDetector(
            arucoDict,
            arucoParams
        )

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            rgb_image, T_C_A = self.pose_estimation(
                rgb_image,
                Detector,
                self.camera_matrix,
                self.dist_coeffs,
                MARKER_SIZE_M
            )
            
            if T_C_A is not None:

                T_A_C = np.linalg.inv(T_C_A)
                self.T_M_C = (self.T_M_A @ T_A_C)
                        
            
            # 1. Run YOLO inference on the RGB frame
            results = self.model.predict(rgb_image, conf=0.6, verbose=False)
            
            # 2. Iterate through detected bounding boxes
            for box in results[0].boxes:
                # Extract integer coordinates of the bounding box window
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure bounding box is within image dimensions to prevent slicing errors
                h, w = rgb_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 3. Calculate the global 2D centroid of the bounding box
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                # 4. Extract the matching "window" (Region of Interest) from the DEPTH array
                depth_window = depth_image[y1:y2, x1:x2]

                # Filter out invalid depth pixels (0.0 or infinite) within the window
                # Assuming kinect depth is in meters (if in mm, change 0.05 to 50, etc.)
                valid_depths = depth_window[(depth_window > 0.05) & np.isfinite(depth_window)]

                if len(valid_depths) > 0:
                    # 5. Use the MEDIAN depth of the window for robust 3D localization
                    # This ignores sensor noise, reflections, and gaps in the bolt
                    z_depth = float(np.median(valid_depths))
                    
                    # 6. Pinhole Camera Projection (2D pixel -> 3D coordinate)
                    point_x = (u - self.cx) * z_depth / self.fx
                    point_y = (v - self.cy) * z_depth / self.fy
                    point_z = z_depth

                    point_C = np.array([
                        point_x,
                        point_y,
                        point_z,
                        1.0
                    ], dtype=np.float64)

                    if self.T_M_C is not None:

                        point_M_h = self.T_M_C @ point_C
                        point_M = point_M_h[:3]

                        Xm = float(point_M[0])
                        Ym = float(point_M[1])
                        Zm = float(point_M[2])

                        print(
                            f"YOLO point in mockup frame: "
                            f"X={Xm:.3f}, "
                            f"Y={Ym:.3f}, "
                            f"Z={Zm:.3f} m"
                        )

                        bolt_index, matched_point, error = (
                            self.associate_bolt(point_M)
                        )

                        if bolt_index is not None:

                            print(
                                f"Matched Bolt #{bolt_index + 1} | "
                                f"Map position = "
                                f"({matched_point[0]:.3f}, "
                                f"{matched_point[1]:.3f}) m | "
                                f"error = {error:.3f} m"
                            )

                        else:

                            print(
                                f"No valid bolt association. "
                                f"Closest error = {error:.3f} m"
                            )

                    bolt_index, matched_point, error = (self.associate_bolt(point_M))

                    if bolt_index is not None:

                        print(
                            f"Matched Bolt #{bolt_index + 1} | "
                            f"Map position = "
                            f"({matched_point[0]:.3f}, "
                            f"{matched_point[1]:.3f}) m | "
                            f"error = {error:.3f} m"
                        )

                    else:
                        print(f"No valid bolt association. "f"Closest error = {error:.3f} m")
                    
                    # Publish the 3D spatial coordinate
                    spatial_msg = Point()
                    spatial_msg.x = point_x
                    spatial_msg.y = point_y
                    spatial_msg.z = point_z
                    self.target_pub.publish(spatial_msg)
                    
                    # --- Draw UI Visualization ---
                    # Draw YOLO Bounding Box
                    cv.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw Centroid Target
                    cv.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    
                    # Display the 3D Z depth on screen
                    depth_display = f"Z: {point_z:.3f}m"
                    cv.putText(rgb_image, depth_display, (x1, y1 - 10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    
            cv.imshow("YOLO 3D Target Localization", rgb_image)
            cv.waitKey(1)
        
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()