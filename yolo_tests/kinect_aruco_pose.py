import time
import cv2 as cv
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class KinectArucoPose(Node):
    def __init__(self):
        super().__init__('kinect_aruco_pose')

        self.bridge = CvBridge()

        # ROS parameters
        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/k4a/rgb/camera_info')
        self.declare_parameter('marker_size_m', 0.10)
        self.declare_parameter('target_id', -1)

        self.rgb_topic = self.get_parameter(
            'rgb_topic'
        ).get_parameter_value().string_value

        self.camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).get_parameter_value().string_value

        self.marker_size_m = self.get_parameter(
            'marker_size_m'
        ).get_parameter_value().double_value

        # target_id = -1 means accept any DICT_5X5_100 marker.
        self.target_id = self.get_parameter(
            'target_id'
        ).get_parameter_value().integer_value

        # Kinect calibration data from CameraInfo.
        self.camera_matrix = None
        self.dist_coeffs = None

        # Latest ArUco -> camera homogeneous transform.
        self.T_C_A = None

        # ArUco detector.
        self.aruco_dictionary = cv.aruco.getPredefinedDictionary(
            cv.aruco.DICT_5X5_100
        )
        self.aruco_parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(
            self.aruco_dictionary,
            self.aruco_parameters
        )

        # Subscribers.
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.last_pose_log_time = 0.0
        self.intrinsics_logged = False

        self.get_logger().info(
            f'Kinect ArUco node started | '
            f'RGB: {self.rgb_topic} | '
            f'CameraInfo: {self.camera_info_topic} | '
            f'marker size: {self.marker_size_m:.4f} m | '
            f'target ID: {self.target_id}'
        )

    def camera_info_callback(self, msg: CameraInfo):
        # Build the 3x3 intrinsic matrix K from CameraInfo.K.
        self.camera_matrix = np.array(
            msg.k,
            dtype=np.float64
        ).reshape(3, 3)

        # Use Kinect distortion coefficients from CameraInfo.D.
        if len(msg.d) > 0:
            self.dist_coeffs = np.array(
                msg.d,
                dtype=np.float64
            ).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros(
                (5, 1),
                dtype=np.float64
            )

        if not self.intrinsics_logged:
            self.intrinsics_logged = True

            self.get_logger().info(
                'Kinect RGB camera calibration received.'
            )

            self.get_logger().info(
                'Camera matrix K:\n'
                + np.array2string(
                    self.camera_matrix,
                    precision=6
                )
            )

            self.get_logger().info(
                'Distortion coefficients D:\n'
                + np.array2string(
                    self.dist_coeffs.reshape(-1),
                    precision=6
                )
            )

    def estimate_marker_pose(self, marker_corners):
        half = self.marker_size_m / 2.0

        # Corner ordering required by SOLVEPNP_IPPE_SQUARE:
        # top-left, top-right, bottom-right, bottom-left.
        marker_points_3d = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0]
        ], dtype=np.float32)

        image_points = marker_corners.reshape(
            4,
            2
        ).astype(np.float32)

        success, rvec, tvec = cv.solvePnP(
            marker_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv.SOLVEPNP_IPPE_SQUARE
        )

        return success, rvec, tvec

    def rgb_callback(self, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as exc:
            self.get_logger().error(
                f'Failed to convert RGB image: {exc}'
            )
            return

        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

        corners, ids, rejected = self.detector.detectMarkers(
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
                marker_id = int(marker_id)

                # Ignore other markers if a specific reference ID is set.
                if (
                    self.target_id >= 0
                    and marker_id != self.target_id
                ):
                    continue

                success, rvec, tvec = self.estimate_marker_pose(
                    corners[i]
                )

                if not success:
                    continue

                # Draw marker coordinate axes.
                cv.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_size_m * 0.75
                )

                # Build T_C_A: ArUco frame -> Kinect RGB camera frame.
                R_C_A, _ = cv.Rodrigues(rvec)

                T_C_A = np.eye(
                    4,
                    dtype=np.float64
                )

                T_C_A[:3, :3] = R_C_A
                T_C_A[:3, 3] = tvec.reshape(3)

                self.T_C_A = T_C_A

                X = float(tvec[0, 0])
                Y = float(tvec[1, 0])
                Z = float(tvec[2, 0])

                marker_corner = corners[i].reshape(4, 2)
                text_x = int(marker_corner[0, 0])
                text_y = max(
                    25,
                    int(marker_corner[0, 1]) - 15
                )

                cv.putText(
                    frame,
                    f'ID {marker_id}: '
                    f'X={X:.3f} Y={Y:.3f} Z={Z:.3f} m',
                    (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # Limit terminal logging to once per second.
                now = time.time()

                if now - self.last_pose_log_time >= 1.0:
                    self.last_pose_log_time = now

                    self.get_logger().info(
                        f'ArUco ID {marker_id} | '
                        f'camera-frame position: '
                        f'X={X:.3f}, Y={Y:.3f}, Z={Z:.3f} m'
                    )

                    self.get_logger().info(
                        'T_C_A:\n'
                        + np.array2string(
                            self.T_C_A,
                            precision=5,
                            suppress_small=True
                        )
                    )

                if self.target_id >= 0:
                    break

        cv.imshow(
            'Kinect ArUco Pose',
            frame
        )

        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = KinectArucoPose()

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
