
#!/usr/bin/env python3

import cv2
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

import time
import statistics

class KinectSegmentationNode(Node):

    def __init__(self):
        super().__init__('kinect_segmentation_node')

        # ROS 2 parameters
        self.declare_parameter(
            'model_path', '/home/amp_b/models/weights.pt'
        )
        self.declare_parameter(
            'image_topic', '/k4a/rgb/image_raw'
        )
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('inference_fps', 15.0)
        self.declare_parameter('device', 'cpu')

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

        # Load YOLO once
        self.get_logger().info(
            f'Loading YOLO model: {model_path}'
        )

        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.latest_frame = None

        #Benchmark configuration
        self.warmup_frames = 10
        self.benchmark_frames = 100

        self.inference_times = []
        self.processed_frames = 0
        self.benchmark_complete = False

        # Camera-friendly QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to Azure Kinect RGB images
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos
        )

        # Run inference at a controlled maximum rate
        self.timer = self.create_timer(
            1.0 / max(float(fps), 0.1),
            self.process_frame
        )

        cv2.namedWindow(
            'Azure Kinect YOLO Segmentation',
            cv2.WINDOW_NORMAL
        )

        self.get_logger().info(
            f'Subscribed to: {image_topic}'
        )
        self.get_logger().info(
            'YOLO segmentation node started'
        )

    def image_callback(self, msg):
        """Receive latest RGB frame from Kinect."""
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

    
    def process_frame(self):
        """Run segmentation and benchmark inference speed."""

        if self.latest_frame is None:
            cv2.waitKey(1)
            return

        frame = self.latest_frame
        self.latest_frame = None

        # Start timing
        start = time.perf_counter()

        # YOLO instance segmentation
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )

        # Synchronize GPU, if applicable, before stopping timer
        if str(self.device).lower().startswith(('cuda', 'gpu')) or (
            str(self.device).isdigit()
        ):
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Stop timing
        inference_ms = (time.perf_counter() - start) * 1000

        self.processed_frames += 1

        # Exclude warm-up frames
        if self.processed_frames > self.warmup_frames:

            if not self.benchmark_complete:
                self.inference_times.append(inference_ms)

                current_fps = 1000.0 / inference_ms

                self.get_logger().info(
                    f'Inference: {inference_ms:.2f} ms | '
                    f'Equivalent FPS: {current_fps:.2f}'
                )

                if len(self.inference_times) >= self.benchmark_frames:
                    self.print_benchmark_results()
                    self.benchmark_complete = True

        # Visualization is excluded from timing
        results[0].names = {
            **results[0].names,
            0: "Hex Bolt"
        }
        annotated_frame = results[0].plot()

        cv2.imshow(
            'Azure Kinect YOLO Segmentation',
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rclpy.shutdown()


    def print_benchmark_results(self):
        """Print statistical processing-speed results."""

        times = self.inference_times

        avg_ms = statistics.mean(times)
        std_ms = statistics.stdev(times)
        min_ms = min(times)
        max_ms = max(times)

        sorted_times = sorted(times)
        p95_ms = sorted_times[
            int(0.95 * (len(sorted_times) - 1))
        ]

        avg_fps = 1000.0 / avg_ms

        self.get_logger().info(
            '\n'
            '========== YOLO BENCHMARK ==========\n'
            f'Frames evaluated: {len(times)}\n'
            f'Average inference: {avg_ms:.2f} ms\n'
            f'Std deviation:     {std_ms:.2f} ms\n'
            f'Min inference:     {min_ms:.2f} ms\n'
            f'Max inference:     {max_ms:.2f} ms\n'
            f'P95 inference:     {p95_ms:.2f} ms\n'
            f'Equivalent FPS:    {avg_fps:.2f}\n'
            '===================================='
        )

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


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
