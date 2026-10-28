
#!/usr/bin/env python3

import time
import statistics

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


# ============================================================
# HEXAGON ORIENTATION ESTIMATION
# ============================================================

def estimate_hex_angle(mask):
    """
    Estimate the orientation of a hexagonal bolt cap
    using sixth-order rotational symmetry.

    Args:
        mask:
            Binary uint8 mask (255 = bolt, 0 = background).

    Returns:
        angle_deg:
            Orientation in [0, 60), or None.

        center:
            (cx, cy) of the segmented cap.

        quality:
            Sixfold symmetry score in [0, 1].
            This is not a calibrated accuracy estimate.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None, None, 0.0

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < 100:
        return None, None, 0.0

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None, None, 0.0

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    points = contour.reshape(
        -1, 2
    ).astype(np.float64)

    x = points[:, 0] - cx

    # Convert image Y to Cartesian Y
    y = cy - points[:, 1]

    radius = np.hypot(x, y)

    valid = radius > 1e-6

    if np.count_nonzero(valid) < 12:
        return None, None, 0.0

    phi = np.arctan2(
        y[valid],
        x[valid]
    )

    # Sixth-order rotational harmonic
    harmonic = np.mean(
        np.exp(6j * phi)
    )

    quality = float(
        np.abs(harmonic)
    )

    angle_deg = (
        np.degrees(np.angle(harmonic)) / 6.0
    ) % 60.0

    return (
        float(angle_deg),
        (float(cx), float(cy)),
        quality
    )


def hex_angle_difference(current, previous):
    """
    Calculate shortest signed difference between two
    hexagon orientations.

    Output range: [-30, 30) degrees.

    Positive = counterclockwise in Cartesian coordinates.
    """

    return (
        (current - previous + 30.0) % 60.0
    ) - 30.0


# ============================================================
# ROS 2 NODE
# ============================================================

class KinectSegmentationNode(Node):

    def __init__(self):

        super().__init__(
            'kinect_segmentation_node'
        )

        # ----------------------------------------------------
        # ROS 2 PARAMETERS
        # ----------------------------------------------------

        self.declare_parameter(
            'model_path',
            '/home/amp_b/models/weights.pt'
        )

        self.declare_parameter(
            'image_topic',
            '/k4a/rgb/image_raw'
        )

        self.declare_parameter(
            'confidence',
            0.5
        )

        self.declare_parameter(
            'image_size',
            640
        )

        self.declare_parameter(
            'inference_fps',
            15.0
        )

        self.declare_parameter(
            'device',
            'cpu'
        )

        self.declare_parameter(
            'max_tracking_distance',
            40.0
        )

        self.declare_parameter(
            'min_hex_quality',
            0.03
        )

        # ----------------------------------------------------
        # READ PARAMETERS
        # ----------------------------------------------------

        model_path = self.get_parameter(
            'model_path'
        ).value

        image_topic = self.get_parameter(
            'image_topic'
        ).value

        self.confidence = self.get_parameter(
            'confidence'
        ).value

        self.image_size = self.get_parameter(
            'image_size'
        ).value

        fps = self.get_parameter(
            'inference_fps'
        ).value

        self.device = self.get_parameter(
            'device'
        ).value

        self.max_tracking_distance = float(
            self.get_parameter(
                'max_tracking_distance'
            ).value
        )

        self.min_hex_quality = float(
            self.get_parameter(
                'min_hex_quality'
            ).value
        )

        # ----------------------------------------------------
        # LOAD YOLO MODEL
        # ----------------------------------------------------

        self.get_logger().info(
            f'Loading YOLO model: {model_path}'
        )

        self.model = YOLO(
            model_path
        )

        self.bridge = CvBridge()

        self.latest_frame = None

        # ----------------------------------------------------
        # BENCHMARK CONFIGURATION
        # ----------------------------------------------------

        self.warmup_frames = 10

        self.benchmark_frames = 100

        self.inference_times = []

        self.processed_frames = 0

        self.benchmark_complete = False

        # ----------------------------------------------------
        # BOLT TRACKING
        # ----------------------------------------------------

        # bolt_id -> (cx, cy)
        self.bolt_centers = {}

        # bolt_id -> baseline angle
        self.bolt_baselines = {}

        # bolt_id -> previous measured angle
        self.previous_angles = {}

        # bolt_id -> cumulative unwrapped angle
        self.total_rotations = {}

        # Unique ID counter
        self.next_bolt_id = 0

        # ----------------------------------------------------
        # CAMERA SUBSCRIBER
        # ----------------------------------------------------

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = (
            self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                qos
            )
        )

        # ----------------------------------------------------
        # INFERENCE TIMER
        # ----------------------------------------------------

        self.timer = self.create_timer(
            1.0 / max(float(fps), 0.1),
            self.process_frame
        )

        # ----------------------------------------------------
        # OPENCV WINDOW
        # ----------------------------------------------------

        self.window_name = (
            'Azure Kinect YOLO Hex Bolt Rotation'
        )

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        self.get_logger().info(
            f'Subscribed to: {image_topic}'
        )

        self.get_logger().info(
            'Hex bolt rotation estimation started'
        )

        self.get_logger().info(
            'Controls: b = reset baselines, q = quit'
        )

    # ========================================================
    # IMAGE CALLBACK
    # ========================================================

    def image_callback(self, msg):
        """
        Receive latest Kinect RGB frame.
        """

        try:

            self.latest_frame = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8'
                )
            )

        except Exception as e:

            self.get_logger().error(
                f'Image conversion failed: {e}'
            )

    # ========================================================
    # BOLT ASSOCIATION
    # ========================================================

    def match_bolt(self, center, used_ids):
        """
        Match a detected bolt to an existing bolt
        using nearest-neighbor center distance.

        Assumes camera and bolts remain mostly stationary.
        """

        cx, cy = center

        best_id = None

        best_distance = (
            self.max_tracking_distance
        )

        for bolt_id, old_center in (
            self.bolt_centers.items()
        ):

            if bolt_id in used_ids:
                continue

            distance = np.hypot(
                cx - old_center[0],
                cy - old_center[1]
            )

            if distance < best_distance:

                best_distance = distance

                best_id = bolt_id

        if best_id is None:

            best_id = self.next_bolt_id

            self.next_bolt_id += 1

            self.get_logger().info(
                f'New bolt detected: ID {best_id}'
            )

        self.bolt_centers[best_id] = center

        return best_id

    # ========================================================
    # BASELINE RESET
    # ========================================================

    def reset_baselines(self):
        """
        Clear saved angles so the next valid
        observation becomes the new baseline.

        Keeps existing bolt IDs and positions.
        """

        self.bolt_baselines.clear()

        self.previous_angles.clear()

        self.total_rotations.clear()

        self.get_logger().info(
            'All bolt rotation baselines reset.'
        )

    # ========================================================
    # ROTATION UPDATE
    # ========================================================

    def update_rotation(self, bolt_id, angle):
        """
        Calculate:

        1. Baseline-relative rotation modulo 60 degrees.
        2. Cumulative rotation using angle unwrapping.

        Angle unwrapping assumes the bolt moves less
        than 30 degrees between valid measurements.
        """

        if bolt_id not in self.bolt_baselines:

            self.bolt_baselines[bolt_id] = angle

            self.previous_angles[bolt_id] = angle

            self.total_rotations[bolt_id] = 0.0

            self.get_logger().info(
                f'Bolt {bolt_id} baseline: '
                f'{angle:.2f} degrees'
            )

        baseline = self.bolt_baselines[
            bolt_id
        ]

        relative_rotation = (
            hex_angle_difference(
                angle,
                baseline
            )
        )

        previous_angle = (
            self.previous_angles[bolt_id]
        )

        delta = hex_angle_difference(
            angle,
            previous_angle
        )

        self.total_rotations[bolt_id] += delta

        self.previous_angles[bolt_id] = angle

        cumulative_rotation = (
            self.total_rotations[bolt_id]
        )

        return (
            relative_rotation,
            cumulative_rotation
        )

    # ========================================================
    # DRAW ORIENTATION
    # ========================================================

    def draw_bolt_info(
        self,
        image,
        bolt_id,
        center,
        angle,
        relative_rotation,
        cumulative_rotation,
        quality
    ):
        """
        Draw center, hexagon orientation, and
        rotation measurements onto the image.
        """

        cx, cy = map(
            int,
            center
        )

        # Center point
        cv2.circle(
            image,
            (cx, cy),
            4,
            (0, 255, 0),
            -1
        )

        # Orientation arrow
        line_length = 35

        theta = np.radians(
            angle
        )

        x2 = int(
            cx + line_length * np.cos(theta)
        )

        y2 = int(
            cy - line_length * np.sin(theta)
        )

        cv2.arrowedLine(
            image,
            (cx, cy),
            (x2, y2),
            (0, 255, 0),
            2,
            tipLength=0.25
        )

        # Text labels
        label1 = (
            f'ID {bolt_id} '
            f'Rel: {relative_rotation:+.1f} deg'
        )

        label2 = (
            f'Total: {cumulative_rotation:+.1f} deg'
        )

        label3 = (
            f'Hex Q: {quality:.2f}'
        )

        text_x = cx + 12
        text_y = cy - 22

        cv2.putText(
            image,
            label1,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2
        )

        cv2.putText(
            image,
            label2,
            (text_x, text_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2
        )

        cv2.putText(
            image,
            label3,
            (text_x, text_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )

    # ========================================================
    # MAIN PROCESSING LOOP
    # ========================================================

    def process_frame(self):
        """
        Run YOLO segmentation, estimate bolt
        orientation, update rotations, and display.

        Benchmark timing covers YOLO inference only.
        """

        if self.latest_frame is None:

            self.handle_keyboard()

            return

        frame = self.latest_frame

        self.latest_frame = None

        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )

        # Synchronize CUDA if applicable
        device_name = str(
            self.device
        ).lower()

        if (
            device_name.startswith(
                ('cuda', 'gpu')
            )
            or device_name.isdigit()
        ):

            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # ----------------------------------------------------
        # BENCHMARK
        # ----------------------------------------------------

        self.processed_frames += 1

        if (
            self.processed_frames
            > self.warmup_frames
        ):

            if not self.benchmark_complete:

                self.inference_times.append(
                    inference_ms
                )

                current_fps = (
                    1000.0 / inference_ms
                )

                self.get_logger().info(
                    f'Inference: {inference_ms:.2f} ms | '
                    f'Equivalent FPS: {current_fps:.2f}'
                )

                if (
                    len(self.inference_times)
                    >= self.benchmark_frames
                ):

                    self.print_benchmark_results()

                    self.benchmark_complete = True

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        result = results[0]

        result.names = {
            **result.names,
            0: 'Hex Bolt'
        }

        annotated_frame = (
            result.plot()
        )

        used_ids = set()

        # ----------------------------------------------------
        # PROCESS EACH INSTANCE MASK
        # ----------------------------------------------------

        if result.masks is not None:

            for polygon in result.masks.xy:

                if len(polygon) < 3:
                    continue

                # Convert polygon into binary mask
                binary_mask = np.zeros(
                    frame.shape[:2],
                    dtype=np.uint8
                )

                polygon_pixels = (
                    np.round(polygon)
                    .astype(np.int32)
                )

                cv2.fillPoly(
                    binary_mask,
                    [polygon_pixels],
                    255
                )

                # Estimate hexagon orientation
                (
                    angle,
                    center,
                    quality
                ) = estimate_hex_angle(
                    binary_mask
                )

                if angle is None:
                    continue

                # Associate with existing bolt
                bolt_id = self.match_bolt(
                    center,
                    used_ids
                )

                used_ids.add(bolt_id)

                # ------------------------------------------------
                # ANGLE QUALITY CHECK
                # ------------------------------------------------

                if quality < self.min_hex_quality:

                    cx, cy = map(
                        int,
                        center
                    )

                    cv2.putText(
                        annotated_frame,
                        f'ID {bolt_id}: Low quality',
                        (cx + 12, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 165, 255),
                        2
                    )

                    continue

                # ------------------------------------------------
                # UPDATE ROTATION
                # ------------------------------------------------

                (
                    relative_rotation,
                    cumulative_rotation
                ) = self.update_rotation(
                    bolt_id,
                    angle
                )

                # ------------------------------------------------
                # DRAW MEASUREMENTS
                # ------------------------------------------------

                self.draw_bolt_info(
                    annotated_frame,
                    bolt_id,
                    center,
                    angle,
                    relative_rotation,
                    cumulative_rotation,
                    quality
                )

        # ----------------------------------------------------
        # SCREEN INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            'b: Reset Baseline | q: Quit',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f'Inference: {inference_ms:.1f} ms',
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            self.window_name,
            annotated_frame
        )

        self.handle_keyboard()

    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    def handle_keyboard(self):

        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):

            self.reset_baselines()

        elif key == ord('q'):

            if rclpy.ok():
                rclpy.shutdown()

    # ========================================================
    # BENCHMARK RESULTS
    # ========================================================

    def print_benchmark_results(self):
        """
        Print statistics for YOLO inference only.
        """

        times = self.inference_times

        if not times:
            return

        avg_ms = statistics.mean(
            times
        )

        std_ms = (
            statistics.stdev(times)
            if len(times) > 1
            else 0.0
        )

        min_ms = min(times)

        max_ms = max(times)

        sorted_times = sorted(
            times
        )

        p95_ms = sorted_times[
            int(
                0.95 * (
                    len(sorted_times) - 1
                )
            )
        ]

        avg_fps = (
            1000.0 / avg_ms
        )

        self.get_logger().info(
            '\n'
            '========== YOLO BENCHMARK ==========\n'
            f'Frames evaluated: {len(times)}\n'
            f'Average inference: {avg_ms:.2f} ms\n'
            f'Std deviation:     {std_ms:.2f} ms\n'
            f'Min inference:    {min_ms:.2f} ms\n'
            f'Max inference:    {max_ms:.2f} ms\n'
            f'P95 inference:    {p95_ms:.2f} ms\n'
            f'Equivalent FPS:   {avg_fps:.2f}\n'
            '===================================='
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(self):

        cv2.destroyAllWindows()

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = KinectSegmentationNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':

    main()
