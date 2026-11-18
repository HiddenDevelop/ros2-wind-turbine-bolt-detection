"""ROS 2 RGB-D bolt localization in an ArUco frame, with XY accuracy diagnostics.

Assumptions:
- rgb/image_raw is the unrectified COLOR image described by rgb/camera_info.
- depth_to_rgb/image_raw contains depth (optical-axis Z) registered to color pixels.
- A detected bolt's bounding-box center represents the *point you intend to measure*.
- One known ArUco marker (default ID 0), printed at the specified true black-square size.

Coordinates published on /hex_bolt/spatial_coordinate are in the ArUco frame.
The normal on /hex_bolt/normal_aruco is +Z in that same frame.
"""

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
        self.declare_parameter('model_path', 'best.pt')
        self.model = YOLO(str(self.get_parameter('model_path').value))

        self.camera_matrix = None
        self.dist_coeffs = None
        self.info_width = None
        self.info_height = None
        self.last_print_time = 0.0
        self.print_interval = 1.0
        self.last_info_warning = 0.0

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/k4a/rgb/camera_info')
        self.declare_parameter('marker_size_m', 0.1778)
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('normal_length_m', 0.08)
        self.declare_parameter('depth_patch_radius_px', 5)
        self.declare_parameter('min_valid_depth_pixels', 5)
        self.declare_parameter('max_marker_reprojection_error_px', 3.0)
        self.declare_parameter('refine_marker_corners', True)
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.normal_length_m = float(self.get_parameter('normal_length_m').value)
        self.patch_radius = int(self.get_parameter('depth_patch_radius_px').value)
        self.min_valid_pixels = int(self.get_parameter('min_valid_depth_pixels').value)
        self.max_marker_rmse = float(self.get_parameter('max_marker_reprojection_error_px').value)
        self.refine_corners = bool(self.get_parameter('refine_marker_corners').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        if (self.marker_size_m <= 0 or self.normal_length_m <= 0 or
                self.patch_radius < 0 or self.min_valid_pixels < 1 or
                self.max_marker_rmse <= 0 or not (0 < self.min_depth_m < self.max_depth_m)):
            raise ValueError('Invalid marker, arrow, depth sampling, or depth limit parameters')

        self.normal_A = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
        parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(dictionary, parameters)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.rgb_sub = Subscriber(self, Image, rgb_topic, qos_profile=image_qos)
        self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=image_qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=5, slop=0.03,
        )
        self.sync.registerCallback(self.kinect_callback)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, 10,
        )
        self.target_pub = self.create_publisher(
            Point, '/hex_bolt/spatial_coordinate', 10,
        )
        self.normal_pub = self.create_publisher(
            Vector3, '/hex_bolt/normal_aruco', 10,
        )
        self.get_logger().info('ArUco localization with XY diagnostics started.')

    def camera_info_callback(self, msg):
        # Use K and D for RAW (distorted) RGB pixels. For rectified RGB, this
        # function and the image topics need to be changed together.
        if msg.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
            self.get_logger().warn(
                f'Unsupported distortion model: {msg.distortion_model}; '
                'OpenCV undistortPoints may not be valid for this camera.'
            )
            return
        if msg.k[0] <= 0 or msg.k[4] <= 0:
            return
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = (
            np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)
            if len(msg.d) else np.zeros((5, 1), dtype=np.float64)
        )
        self.info_width = int(msg.width)
        self.info_height = int(msg.height)

    def estimate_aruco_pose(self, raw_frame, display_frame):
        """Return camera-to-marker matrix and corner reprojection RMSE in pixels."""
        gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None, None

        for i, detected_id in enumerate(ids.flatten()):
            if int(detected_id) != self.marker_id:
                continue

            image_points = corners[i].reshape(4, 2).astype(np.float32)
            if self.refine_corners:
                # Refine the four corner positions to sub-pixel precision.
                refined = image_points.reshape(-1, 1, 2).copy()
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                            40, 0.001)
                cv.cornerSubPix(gray, refined, (5, 5), (-1, -1), criteria)
                image_points = refined.reshape(4, 2)

            cv.aruco.drawDetectedMarkers(
                display_frame, [image_points.reshape(1, 4, 2)],
                np.array([[self.marker_id]], dtype=np.int32),
            )
            half = self.marker_size_m / 2.0
            marker_points = np.array([
                [-half, half, 0.0], [half, half, 0.0],
                [half, -half, 0.0], [-half, -half, 0.0],
            ], dtype=np.float64)
            success, rvec, tvec = cv.solvePnP(
                marker_points, image_points.astype(np.float64),
                self.camera_matrix, self.dist_coeffs,
                flags=cv.SOLVEPNP_IPPE_SQUARE,
            )
            if not success or float(tvec[2, 0]) <= 0:
                return None, None

            # This measures consistency of the *observed corners* with the
            # fitted pose. Low RMSE does NOT prove true metric accuracy.
            projected, _ = cv.projectPoints(
                marker_points, rvec, tvec,
                self.camera_matrix, self.dist_coeffs,
            )
            residual = projected.reshape(4, 2) - image_points
            rmse_px = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
            if rmse_px > self.max_marker_rmse:
                cv.putText(display_frame, f'Marker RMSE {rmse_px:.2f}px: REJECTED',
                           (15, 35), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                           (0, 0, 255), 2)
                return None, rmse_px

            cv.drawFrameAxes(
                display_frame, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size_m * 0.75,
            )
            R_C_A, _ = cv.Rodrigues(rvec)
            T_C_A = np.eye(4, dtype=np.float64)
            T_C_A[:3, :3] = R_C_A
            T_C_A[:3, 3] = tvec.reshape(3)
            return np.linalg.inv(T_C_A), rmse_px
        return None, None

    def get_depth_at_target(self, depth_image, u, v, bbox):
        """Median depth in a center patch contained within the YOLO box."""
        x1, y1, x2, y2 = bbox
        h, w = depth_image.shape[:2]
        r = self.patch_radius
        left = max(x1, 0, u - r)
        right = min(x2, w, u + r + 1)
        top = max(y1, 0, v - r)
        bottom = min(y2, h, v + r + 1)
        if left >= right or top >= bottom:
            return None, 0, None
        patch = depth_image[top:bottom, left:right]
        depths = patch.astype(np.float64)
        if np.issubdtype(patch.dtype, np.integer):
            depths *= 0.001  # Azure Kinect 16UC1 depth: millimeters to meters
        valid = depths[(depths >= self.min_depth_m) &
                       (depths <= self.max_depth_m) & np.isfinite(depths)]
        if valid.size < self.min_valid_pixels:
            return None, int(valid.size), None
        # IQR helps identify a patch crossing a depth boundary.
        iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25))
        return float(np.median(valid)), int(valid.size), iqr

    def kinect_callback(self, rgb_msg, depth_msg):
        if self.camera_matrix is None:
            return
        try:
            raw_rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg, desired_encoding='bgr8',
            )
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough',
            )
            if depth_image.ndim != 2 or depth_image.shape != raw_rgb.shape[:2]:
                self.get_logger().warn('Registered depth and RGB pixel grids differ; skipping.')
                return
            h, w = raw_rgb.shape[:2]
            if (w, h) != (self.info_width, self.info_height):
                self.get_logger().warn('RGB image and camera_info dimensions differ; skipping.')
                return

            # Always detect and estimate from an UNANNOTATED image.
            display = raw_rgb.copy()
            T_A_C, marker_rmse = self.estimate_aruco_pose(raw_rgb, display)
            if T_A_C is None:
                cv.putText(display, 'Reference ArUco pose unavailable', (15, 65),
                           cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv.imshow('YOLO 3D Target Localization', display)
                cv.waitKey(1)
                return

            results = self.model.predict(raw_rgb, conf=0.3, verbose=False)
            now = time.monotonic()
            should_print = (now - self.last_print_time) >= self.print_interval
            R_A_C = T_A_C[:3, :3]
            normal_C = R_A_C.T @ self.normal_A

            cv.putText(display, f'ArUco reprojection RMSE: {marker_rmse:.2f} px',
                       (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 2)

            for detection_id, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                z_depth, valid_count, depth_iqr = self.get_depth_at_target(
                    depth_image, u, v, (x1, y1, x2, y2),
                )
                cv.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.circle(display, (u, v), 4, (0, 0, 255), -1)
                if z_depth is None:
                    cv.putText(display, f'D{detection_id}: no center depth',
                               (x1, max(20, y1 - 9)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    continue

                # The raw RGB pixel must be undistorted BEFORE back-projecting.
                # Output is normalized ray (x_norm, y_norm), not a pixel.
                raw_pixel = np.array([[[float(u), float(v)]]], dtype=np.float64)
                normalized = cv.undistortPoints(
                    raw_pixel, self.camera_matrix, self.dist_coeffs,
                )[0, 0]
                point_C_xyz = np.array([
                    normalized[0] * z_depth,
                    normalized[1] * z_depth,
                    z_depth,
                ], dtype=np.float64)
                point_C = np.r_[point_C_xyz, 1.0]
                point_A = (T_A_C @ point_C)[:3]
                Xa, Ya, Za = map(float, point_A)

                arrow_end_C = point_C_xyz + self.normal_length_m * normal_C
                projected, _ = cv.projectPoints(
                    np.vstack((point_C_xyz, arrow_end_C)),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    self.camera_matrix, self.dist_coeffs,
                )
                arrow_start, arrow_end = [
                    tuple(np.rint(p).astype(int)) for p in projected.reshape(-1, 2)
                ]
                cv.arrowedLine(display, arrow_start, arrow_end,
                               (255, 0, 255), 3, cv.LINE_AA, tipLength=0.25)
                cv.putText(display, '',
                           (arrow_end[0] + 6, arrow_end[1] - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                cv.putText(display, f'D{detection_id} A: ({Xa:.3f}, {Ya:.3f}, {Za:.3f})m',
                           (x1, max(20, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX,
                           0.45, (0, 255, 0), 2)

                position = Point(x=Xa, y=Ya, z=Za)
                self.target_pub.publish(position)
                self.normal_pub.publish(Vector3(x=0.0, y=0.0, z=1.0))
                if should_print:
                    print(f'[D{detection_id}] ArUco XYZ=({Xa:.4f}, {Ya:.4f}, {Za:.4f}) m; '
                          f'pixel=({u}, {v}); camera Z={z_depth:.4f} m; '
                          f'center_depth_pixels={valid_count}; '
                          f'depth_IQR={depth_iqr * 1000:.1f} mm; '
                          f'marker_RMSE={marker_rmse:.2f} px')
            if should_print:
                self.last_print_time = now
            cv.imshow('YOLO 3D Target Localization', display)
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
