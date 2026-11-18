"""ROS 2 YOLO RGB-D localization in an ArUco marker coordinate frame."""

import time

import cv2 as cv
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO


class YoloDepthLocalization(Node):
    def __init__(self):
        super().__init__('yolo_depth_localization')

        self.bridge = CvBridge()
        self.model = YOLO('best.pt')

        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = self.fy = self.cx = self.cy = None
        self.last_print_time = 0.0
        self.print_interval = 1.0

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/k4a/rgb/camera_info')
        self.declare_parameter('marker_size_m', 0.1778)
        self.declare_parameter('marker_id', 0)  # Change to the ID on your physical marker.
        self.declare_parameter('normal_length_m', 0.08)  # Arrow display length, meters.

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.normal_length_m = float(self.get_parameter('normal_length_m').value)
        if self.normal_length_m <= 0:
            raise ValueError('normal_length_m must be positive')

        # Unit normal to the ArUco XY plane, in the +Z direction.
        self.normal_A = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
        self.detector = cv.aruco.ArucoDetector(
            aruco_dictionary, cv.aruco.DetectorParameters()
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.rgb_sub = Subscriber(self, Image, rgb_topic, qos_profile=image_qos)
        self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=image_qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=2, slop=0.1
        )
        self.sync.registerCallback(self.kinect_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, 10
        )
        # Published Point is in the ArUco marker frame, units: meters.
        self.target_pub = self.create_publisher(
            Point, '/hex_bolt/spatial_coordinate', 10
        )
        self.normal_pub = self.create_publisher(
            Vector3, '/hex_bolt/normal_aruco', 10
        )
        self.get_logger().info('YOLO ArUco-frame 3D localization started.')

    def camera_info_callback(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])
        self.dist_coeffs = (
            np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)
            if len(msg.d) else np.zeros((5, 1), dtype=np.float64)
        )

    def estimate_aruco_pose(self, frame):
        """Return T_A_C: transformation from camera frame C to marker frame A."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None

        for i, detected_id in enumerate(ids.flatten()):
            if int(detected_id) != self.marker_id:
                continue

            cv.aruco.drawDetectedMarkers(frame, [corners[i]], ids[i:i + 1])
            half = self.marker_size_m / 2.0
            marker_points = np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)
            image_points = corners[i].reshape(4, 2).astype(np.float32)
            success, rvec, tvec = cv.solvePnP(
                marker_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                return None

            cv.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size_m * 0.75
            )

            # solvePnP provides marker -> camera; invert to get camera -> marker.
            R_C_A, _ = cv.Rodrigues(rvec)
            T_C_A = np.eye(4, dtype=np.float64)
            T_C_A[:3, :3] = R_C_A
            T_C_A[:3, 3] = tvec.reshape(3)
            return np.linalg.inv(T_C_A)

        return None

    def kinect_callback(self, rgb_msg, depth_msg):
        if self.camera_matrix is None:
            return
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )

            # Both arrays must represent the same aligned pixel grid.
            if depth_image.shape[:2] != rgb_image.shape[:2]:
                self.get_logger().warn('RGB and depth dimensions differ; skipping frame.')
                return

            T_A_C = self.estimate_aruco_pose(rgb_image)
            if T_A_C is None:
                cv.putText(rgb_image, 'Reference ArUco marker not detected',
                           (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.imshow('YOLO 3D Target Localization', rgb_image)
                cv.waitKey(1)
                return

            results = self.model.predict(rgb_image, conf=0.3, verbose=False)
            now = time.time()
            should_print = (now - self.last_print_time) >= self.print_interval

            h, w = rgb_image.shape[:2]
            for detection_id, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                depth_window = depth_image[y1:y2, x1:x2]

                # Azure Kinect commonly publishes uint16 depth in millimeters.
                # Floating-point depth images are assumed to contain meters.
                depth_m = depth_window.astype(np.float64)
                if np.issubdtype(depth_window.dtype, np.integer):
                    depth_m *= 0.001
                valid_depths = depth_m[(depth_m > 0.05) & np.isfinite(depth_m)]
                if valid_depths.size == 0:
                    continue

                z_depth = float(np.median(valid_depths))
                point_C = np.array([
                    (u - self.cx) * z_depth / self.fx,
                    (v - self.cy) * z_depth / self.fy,
                    z_depth,
                    1.0,
                ], dtype=np.float64)

                # Camera coordinates -> ArUco marker coordinates.
                point_A = (T_A_C @ point_C)[:3]
                Xa, Ya, Za = map(float, point_A)

                # All bolts share the ArUco plane's +Z unit normal in frame A.
                # Map it to camera coordinates for 2D visualization.
                R_A_C = T_A_C[:3, :3]
                normal_C = R_A_C.T @ self.normal_A
                arrow_end_C = point_C[:3] + self.normal_length_m * normal_C

                # projectPoints accounts for camera distortion when drawing.
                pixels, _ = cv.projectPoints(
                    np.vstack((point_C[:3], arrow_end_C)),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    self.camera_matrix, self.dist_coeffs,
                )
                pixels = pixels.reshape(-1, 2)
                arrow_start = tuple(np.rint(pixels[0]).astype(int))
                arrow_end = tuple(np.rint(pixels[1]).astype(int))

                spatial_msg = Point()
                spatial_msg.x = Xa
                spatial_msg.y = Ya
                spatial_msg.z = Za
                self.target_pub.publish(spatial_msg)

                normal_msg = Vector3()
                normal_msg.x, normal_msg.y, normal_msg.z = map(float, self.normal_A)
                self.normal_pub.publish(normal_msg)

                if should_print:
                    print(f'[D{detection_id}] ArUco-frame position: '
                          f'X={Xa:.3f}, Y={Ya:.3f}, Z={Za:.3f} m; '
                          f'normal_A=(0, 0, +1)')

                cv.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                # Magenta arrow: bolt -> ArUco +Z normal (not camera +Z).
                cv.arrowedLine(
                    rgb_image, arrow_start, arrow_end, (255, 0, 255),
                    thickness=3, tipLength=0.25, line_type=cv.LINE_AA,
                )
                cv.putText(
                    rgb_image, '+Z ArUco', (arrow_end[0] + 6, arrow_end[1] - 6),
                    cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2,
                )
                cv.putText(rgb_image, f'D{detection_id}', (x1, max(15, y1 - 30)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(rgb_image, f'A: ({Xa:.3f}, {Ya:.3f}, {Za:.3f}) m',
                           (x1, max(30, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX,
                           0.45, (0, 255, 0), 2)

            if should_print:
                self.last_print_time = now
            cv.imshow('YOLO 3D Target Localization', rgb_image)
            cv.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f'Processing error: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
