#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class MockupPlaneTracker(Node):
    """
    ROS 2 proof-of-concept for tracking a fixed 1.40 m x 1.40 m
    mockup/world plane in the Azure Kinect RGB image.

    Current goal only:
      1. Define the metric mockup plane.
      2. Manually select its four physical corners once.
      3. Estimate its pose relative to the camera with PnP.
      4. Track those four image points with Lucas-Kanade optical flow.
      5. Re-estimate PnP as the camera view changes.
      6. Project the fixed plane/grid back into the camera FOV.

    This version intentionally does NOT use bolt detections, bolt coordinates,
    depth measurements, or the EKF yet.
    """

    def __init__(self):
        super().__init__('mockup_plane_tracker')

        self.bridge = CvBridge()

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/k4a/rgb/camera_info')

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        camera_info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value
        )

        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.camera_matrix = None
        self.dist_coeffs = None

        # ============================================================
        # 2. FIXED WORLD / MOCKUP GEOMETRY
        # ============================================================
        #
        # world_mockup convention:
        #   origin = bottom-right corner
        #   +X = right, therefore physical plane extends toward negative X
        #   +Y = upward
        #   +Z = normal to plane
        #
        # Corner order:
        #
        #   P2 ---------------- P3
        #   |                   |
        #   |                   |
        #   |                   |
        #   P1 ---------------- P0
        #                        ^
        #                      origin
        #
        # Click image corners in the SAME order.

        self.plane_points_M = np.array([
            [ 0.00, 0.00, 0.00],   # P0: bottom-right / origin
            [-1.40, 0.00, 0.00],   # P1: bottom-left
            [-1.40, 1.40, 0.00],   # P2: top-left
            [ 0.00, 1.40, 0.00],   # P3: top-right
        ], dtype=np.float32)

        self.grid_spacing_m = 0.20

        # LK parameters for cv.calcOpticalFlowPyrLK

        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(
                cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                50,
                0.001
            )
        )

        self.reference_frame = None
        self.reference_gray = None

        self.selected_points_img = []

        self.previous_gray = None
        self.previous_points_img = None

        self.tracking_active = False

        self.rvec = None
        self.tvec = None

        # mockup -> camera
        self.T_C_M = None

        # camera -> mockup
        self.T_M_C = None

        self.window_name = 'Mockup Plane Tracking'

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info(
            'Mockup plane tracker started. Waiting for RGB camera intrinsics...'
        )

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None:
            return

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

        self.get_logger().info('RGB camera intrinsics received.')
        self.get_logger().info(f'K =\n{self.camera_matrix}')

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return

        if self.tracking_active:
            return

        if self.reference_frame is None:
            return

        if len(self.selected_points_img) >= 4:
            return

        self.selected_points_img.append([float(x), float(y)])

        index = len(self.selected_points_img) - 1
        self.get_logger().info(f'Selected P{index}: ({x}, {y})')

    def begin_reference_selection(self, frame, gray, reason=None):
        self.reference_frame = frame.copy()
        self.reference_gray = gray.copy()

        self.selected_points_img = []

        self.previous_gray = None
        self.previous_points_img = None

        self.tracking_active = False

        self.rvec = None
        self.tvec = None
        self.T_C_M = None
        self.T_M_C = None

        if reason is not None:
            self.get_logger().warn(reason)

        self.get_logger().info(
            'Select plane corners in order: '
            'P0 bottom-right/origin -> '
            'P1 bottom-left -> '
            'P2 top-left -> '
            'P3 top-right.'
        )

    def initialize_pose_and_tracking(self):
        image_points = np.array(
            self.selected_points_img,
            dtype=np.float32
        )

        success, rvec, tvec = cv.solvePnP(
            self.plane_points_M,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv.SOLVEPNP_IPPE
        )

        if not success:
            self.get_logger().error(
                'Initial solvePnP failed. Press R and select the plane corners again.'
            )
            return False

        self.rvec = rvec
        self.tvec = tvec

        self.update_transform_matrices()

        self.previous_gray = self.reference_gray.copy()

        self.previous_points_img = image_points.reshape(-1, 1, 2)

        self.tracking_active = True

        self.get_logger().info(
            'Initial plane pose established. LK + PnP tracking active.'
        )

        return True

    # ================================================================
    # TRANSFORM MATH
    # ================================================================

    def update_transform_matrices(self):
        rotation_C_M, _ = cv.Rodrigues(self.rvec)

        self.T_C_M = np.eye(4, dtype=np.float64)
        self.T_C_M[:3, :3] = rotation_C_M
        self.T_C_M[:3, 3] = self.tvec.reshape(3)

        self.T_M_C = np.linalg.inv(self.T_C_M)

    # ================================================================
    # VISUALIZATION
    # ================================================================

    @staticmethod
    def to_int_pixel(point):
        return (
            int(round(float(point[0]))),
            int(round(float(point[1])))
        )

    def draw_selection_screen(self):
        display = self.reference_frame.copy()

        labels = [
            'P0 bottom-right / origin',
            'P1 bottom-left',
            'P2 top-left',
            'P3 top-right'
        ]

        cv.putText(
            display,
            'Select plane corners: P0 BR -> P1 BL -> P2 TL -> P3 TR',
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv.putText(
            display,
            'R = reset selection | Q = quit',
            (20, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        for i, point in enumerate(self.selected_points_img):
            pixel = self.to_int_pixel(point)

            cv.circle(display, pixel, 7, (0, 0, 255), -1)

            cv.putText(
                display,
                f'P{i}',
                (pixel[0] + 8, pixel[1] - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        next_index = len(self.selected_points_img)

        if next_index < 4:
            cv.putText(
                display,
                f'Next: {labels[next_index]}',
                (20, 90),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        return display

    def draw_projected_plane(self, frame):
        # Plane boundary
        projected_corners, _ = cv.projectPoints(
            self.plane_points_M,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        projected_corners = projected_corners.reshape(-1, 2)

        polygon = np.round(
            projected_corners
        ).astype(np.int32).reshape(-1, 1, 2)

        cv.polylines(
            frame,
            [polygon],
            True,
            (0, 255, 0),
            3
        )

        # 20 cm grid
        x_values = np.arange(-1.40, 0.0001, self.grid_spacing_m)
        y_values = np.arange(0.0, 1.4001, self.grid_spacing_m)

        for x_m in x_values:
            line_M = np.array([
                [x_m, 0.0, 0.0],
                [x_m, 1.40, 0.0]
            ], dtype=np.float32)

            line_img, _ = cv.projectPoints(
                line_M,
                self.rvec,
                self.tvec,
                self.camera_matrix,
                self.dist_coeffs
            )

            p0, p1 = line_img.reshape(2, 2)

            cv.line(
                frame,
                self.to_int_pixel(p0),
                self.to_int_pixel(p1),
                (0, 160, 0),
                1
            )

        for y_m in y_values:
            line_M = np.array([
                [0.0, y_m, 0.0],
                [-1.40, y_m, 0.0]
            ], dtype=np.float32)

            line_img, _ = cv.projectPoints(
                line_M,
                self.rvec,
                self.tvec,
                self.camera_matrix,
                self.dist_coeffs
            )

            p0, p1 = line_img.reshape(2, 2)

            cv.line(
                frame,
                self.to_int_pixel(p0),
                self.to_int_pixel(p1),
                (0, 160, 0),
                1
            )

        # Origin and axis directions
        origin = self.to_int_pixel(projected_corners[0])
        bottom_left = self.to_int_pixel(projected_corners[1])
        top_right = self.to_int_pixel(projected_corners[3])

        cv.circle(frame, origin, 7, (0, 0, 255), -1)

        cv.putText(
            frame,
            'O (0,0)',
            (origin[0] + 8, origin[1] - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        cv.arrowedLine(
            frame,
            origin,
            bottom_left,
            (255, 0, 0),
            2,
            tipLength=0.04
        )

        cv.putText(
            frame,
            '-X',
            bottom_left,
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        cv.arrowedLine(
            frame,
            origin,
            top_right,
            (0, 0, 255),
            2,
            tipLength=0.04
        )

        cv.putText(
            frame,
            '+Y',
            top_right,
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    def draw_tracked_reference_points(self, frame, image_points):
        for i, point in enumerate(image_points):
            pixel = self.to_int_pixel(point)

            cv.circle(frame, pixel, 5, (255, 0, 255), -1)

            cv.putText(
                frame,
                f'P{i}',
                (pixel[0] + 7, pixel[1] - 7),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

    # ================================================================
    # MAIN RGB CALLBACK
    # ================================================================

    def rgb_callback(self, rgb_msg):
        if self.camera_matrix is None:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )

            gray = cv.cvtColor(
                rgb_image,
                cv.COLOR_BGR2GRAY
            )

            # First RGB frame becomes the frozen initialization frame.
            if self.reference_frame is None:
                self.begin_reference_selection(rgb_image, gray)

            # --------------------------------------------------------
            # REFERENCE SELECTION MODE
            # --------------------------------------------------------

            if not self.tracking_active:
                display = self.draw_selection_screen()

                if len(self.selected_points_img) == 4:
                    initialized = self.initialize_pose_and_tracking()

                    if initialized:
                        display = self.reference_frame.copy()

                        self.draw_projected_plane(display)

                        self.draw_tracked_reference_points(
                            display,
                            np.array(
                                self.selected_points_img,
                                dtype=np.float32
                            )
                        )

                cv.imshow(self.window_name, display)
                self.handle_keyboard(rgb_image, gray)
                return

            # --------------------------------------------------------
            # LUCAS-KANADE TRACKING
            # --------------------------------------------------------

            current_points, status, error = cv.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.previous_points_img,
                None,
                **self.lk_params
            )

            if current_points is None or status is None:
                self.begin_reference_selection(
                    rgb_image,
                    gray,
                    reason='LK tracking failed. Re-select the four plane corners.'
                )
                return

            status = status.reshape(-1)
            good_count = int(np.sum(status))

            if good_count != 4:
                lost = [
                    f'P{i}'
                    for i, valid in enumerate(status)
                    if valid == 0
                ]

                self.begin_reference_selection(
                    rgb_image,
                    gray,
                    reason=(
                        f'LK lost {4 - good_count} reference point(s): '
                        f'{", ".join(lost)}. Re-select the plane.'
                    )
                )
                return

            image_points = current_points.reshape(4, 2).astype(np.float32)

            # --------------------------------------------------------
            # UPDATE PNP POSE
            # --------------------------------------------------------

            rvec_guess = self.rvec.copy()
            tvec_guess = self.tvec.copy()

            success, rvec, tvec = cv.solvePnP(
                self.plane_points_M,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                rvec_guess,
                tvec_guess,
                True,
                cv.SOLVEPNP_ITERATIVE
            )

            if not success:
                self.begin_reference_selection(
                    rgb_image,
                    gray,
                    reason='solvePnP failed. Re-select the four plane corners.'
                )
                return

            self.rvec = rvec
            self.tvec = tvec

            self.update_transform_matrices()

            # --------------------------------------------------------
            # DRAW FIXED WORLD PLANE INTO CURRENT CAMERA VIEW
            # --------------------------------------------------------

            display = rgb_image.copy()

            self.draw_projected_plane(display)
            self.draw_tracked_reference_points(display, image_points)

            cv.putText(
                display,
                'Plane tracking active | R = reset | Q = quit',
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv.putText(
                display,
                'Green = projected 1.40 m x 1.40 m world_mockup plane',
                (20, 58),
                cv.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2
            )

            camera_position_M = self.T_M_C[:3, 3]

            cv.putText(
                display,
                (
                    'Camera in mockup frame: '
                    f'X={camera_position_M[0]:.3f} '
                    f'Y={camera_position_M[1]:.3f} '
                    f'Z={camera_position_M[2]:.3f} m'
                ),
                (20, 86),
                cv.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv.imshow(self.window_name, display)

            # Update LK state for next frame.
            self.previous_gray = gray.copy()
            self.previous_points_img = current_points.copy()

            self.handle_keyboard(rgb_image, gray)

        except Exception as exc:
            self.get_logger().error(f'RGB processing error: {exc}')

    # ================================================================
    # KEYBOARD CONTROLS
    # ================================================================

    def handle_keyboard(self, current_frame, current_gray):
        key = cv.waitKey(1) & 0xFF

        if key == ord('r'):
            self.begin_reference_selection(
                current_frame,
                current_gray,
                reason='Reference-plane selection reset by user.'
            )

        elif key == ord('q'):
            self.get_logger().info('Quit requested.')

            if rclpy.ok():
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = MockupPlaneTracker()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
