#!/usr/bin/env python3

import time
import statistics

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge


class KinectSensor(Node):

    def __init__(self):
        super().__init__('kinect_sensor')

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter(
            'depth_topic', '/k4a/depth_to_rgb/image_raw'
        )

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value

        # Benchmark configuration
        self.warmup_frames = 10
        self.benchmark_frames = 100

        self.processing_times = []
        self.processed_frames = 0
        self.benchmark_complete = False

        # Reuse OpenCV objects
        self.clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (7, 7)
        )

        # RGB + depth synchronization
        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

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

        self.get_logger().info(
            'Classical CV benchmark started'
        )

    def auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        return cv2.Canny(image, lower, upper)

    def get_internal_angles(self, approx):
        angles = []

        for i in range(6):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 6][0]
            p0 = approx[(i - 1) % 6][0]

            v1 = p0 - p1
            v2 = p2 - p1

            cosine_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) *
                np.linalg.norm(v2) + 1e-5
            )

            angle = np.degrees(
                np.arccos(
                    np.clip(cosine_angle, -1.0, 1.0)
                )
            )

            angles.append(angle)

        return angles

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def kinect_callback(self, rgb_msg, depth_msg):

        try:
            # Image conversion excluded from benchmark
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )

            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding='passthrough'
            )

            # ======================================
            # START PROCESSING TIMER
            # ======================================

            start = time.perf_counter()

            # Depth range: 10 cm to 60 cm
            if depth_image.dtype == np.uint16:
                min_depth = 100
                max_depth = 600
            else:
                min_depth = 0.100
                max_depth = 0.600

            depth_valid = (
                np.isfinite(depth_image) &
                (depth_image >= min_depth) &
                (depth_image <= max_depth)
            )

            mask = np.uint8(depth_valid) * 255

            isolated_rgb = cv2.bitwise_and(
                rgb_image,
                rgb_image,
                mask=mask
            )

            gray = cv2.cvtColor(
                isolated_rgb,
                cv2.COLOR_BGR2GRAY
            )

            enhanced_gray = self.clahe.apply(gray)

            blurred = cv2.bilateralFilter(
                enhanced_gray,
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            )

            edged = self.auto_canny(blurred)

            closed = cv2.morphologyEx(
                edged,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=2
            )

            contours, _ = cv2.findContours(
                closed.copy(),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Store detections without drawing
            detections = []

            for cnt in contours:

                peri = cv2.arcLength(cnt, True)

                if peri < 80:
                    continue

                epsilon = 0.045 * peri

                approx = cv2.approxPolyDP(
                    cnt,
                    epsilon,
                    True
                )

                if (
                    len(approx) == 6
                    and cv2.isContourConvex(approx)
                ):

                    angles = self.get_internal_angles(
                        approx
                    )

                    if (
                        min(angles) > 70
                        and max(angles) < 170
                    ):

                        x, y, w, h = cv2.boundingRect(
                            approx
                        )

                        aspect_ratio = float(w) / h

                        if 0.4 <= aspect_ratio <= 2.5:

                            M = cv2.moments(cnt)

                            if M["m00"] != 0:

                                cX = int(
                                    M["m10"] / M["m00"]
                                )

                                cY = int(
                                    M["m01"] / M["m00"]
                                )

                                detections.append(
                                    (approx, cX, cY)
                                )

            # ======================================
            # STOP PROCESSING TIMER
            # ======================================

            processing_ms = (
                time.perf_counter() - start
            ) * 1000.0

            self.processed_frames += 1

            # Exclude warm-up
            if self.processed_frames > self.warmup_frames:

                if not self.benchmark_complete:

                    self.processing_times.append(
                        processing_ms
                    )

                    if (
                        len(self.processing_times)
                        >= self.benchmark_frames
                    ):
                        self.print_benchmark_results()
                        self.benchmark_complete = True

            # ======================================
            # VISUALIZATION (NOT TIMED)
            # ======================================

            for approx, cX, cY in detections:

                cv2.drawContours(
                    rgb_image,
                    [approx],
                    -1,
                    (0, 255, 0),
                    3
                )

                cv2.circle(
                    rgb_image,
                    (cX, cY),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    rgb_image,
                    "Hex Bolt",
                    (cX - 40, cY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "Robot Hex Localization",
                rgb_image
            )

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(
                f'Kinect processing error: {e}'
            )

    def print_benchmark_results(self):

        times = self.processing_times

        avg_ms = statistics.mean(times)
        std_ms = statistics.stdev(times)
        min_ms = min(times)
        max_ms = max(times)

        p95_ms = float(
            np.percentile(times, 95)
        )

        equivalent_fps = 1000.0 / avg_ms

        self.get_logger().info(
            '\n'
            '======= CLASSICAL CV BENCHMARK =======\n'
            f'Frames evaluated: {len(times)}\n'
            f'Average processing: {avg_ms:.2f} ms\n'
            f'Std deviation:      {std_ms:.2f} ms\n'
            f'Min processing:     {min_ms:.2f} ms\n'
            f'Max processing:     {max_ms:.2f} ms\n'
            f'P95 processing:     {p95_ms:.2f} ms\n'
            f'Equivalent FPS:     {equivalent_fps:.2f}\n'
            '======================================'
        )


def main(args=None):

    rclpy.init(args=args)

    node = KinectSensor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()