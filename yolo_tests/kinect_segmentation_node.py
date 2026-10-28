#!/usr/bin/env python3

import statistics
import time
from collections import deque

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO


class KinectSegmentationNode(Node):
    def __init__(self):
        super().__init__('kinect_segmentation_node')

        self.declare_parameter('model_path', '/home/amp_b/models/weights.pt')
        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('inference_fps', 15.0)
        self.declare_parameter('device', 'cuda:0')

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        self.confidence = float(self.get_parameter('confidence').value)
        self.image_size = int(self.get_parameter('image_size').value)
        target_fps = float(self.get_parameter('inference_fps').value)
        self.device = str(self.get_parameter('device').value)

        self.use_cuda = self.device.lower().startswith(('cuda', 'gpu')) or self.device.isdigit()
        self.get_logger().info(f'PyTorch: {torch.__version__}')
        self.get_logger().info(f'PyTorch CUDA runtime: {torch.version.cuda}')
        self.get_logger().info(f'CUDA available: {torch.cuda.is_available()}')

        if self.use_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f'Device {self.device!r} requested, but PyTorch CUDA is unavailable.'
                )
            self.get_logger().info(
                f'Using CUDA device {self.device}: {torch.cuda.get_device_name(0)}'
            )
        else:
            self.get_logger().info(f'Using inference device: {self.device}')

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.latest_frame = None

        # Exclude startup/warmup iterations from the reported benchmark.
        self.warmup_frames = 10
        self.benchmark_frames = 100
        self.processed_frames = 0
        self.inference_times = []
        self.preprocess_times = []
        self.model_times = []
        self.postprocess_times = []
        self.benchmark_complete = False

        # Track real processed/displayed throughput over a rolling interval.
        self.last_frame_completed_at = None
        self.frame_intervals = deque(maxlen=30)
        self.last_status_log_at = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, qos
        )
        self.timer = self.create_timer(
            1.0 / max(target_fps, 0.1), self.process_frame
        )

        cv2.namedWindow('Azure Kinect YOLO Segmentation', cv2.WINDOW_NORMAL)
        self.get_logger().info(f'Subscribed to: {image_topic}')
        self.get_logger().info(f'Inference timer limit: {target_fps:.1f} FPS')
        self.get_logger().info('YOLO segmentation node started')

    def image_callback(self, msg):
        start = time.perf_counter()
    
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
    
            conversion_ms = (
                time.perf_counter() - start
            ) * 1000
    
           
    
        except Exception as e:
            self.get_logger().error(
                f'Image conversion failed: {e}'
            )

    def process_frame(self):
        """Run segmentation, benchmark predict(), and show annotated output."""
        if self.latest_frame is None:
            cv2.waitKey(1)
            return

        frame = self.latest_frame
        self.latest_frame = None

        # CUDA kernels are asynchronous; synchronize around the wall-clock timer.
        if self.use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if self.use_cuda:
            torch.cuda.synchronize()
        predict_ms = (time.perf_counter() - start) * 1000.0

        self.processed_frames += 1
        result = results[0]
        speed = result.speed or {}

        if self.processed_frames > self.warmup_frames and not self.benchmark_complete:
            self.inference_times.append(predict_ms)
            self.preprocess_times.append(float(speed.get('preprocess', 0.0)))
            self.model_times.append(float(speed.get('inference', 0.0)))
            self.postprocess_times.append(float(speed.get('postprocess', 0.0)))
            if len(self.inference_times) >= self.benchmark_frames:
                self.print_benchmark_results()
                self.benchmark_complete = True

        # Visualization is deliberately excluded from the predict() benchmark.
        result.names = {**result.names, 0: 'Hex Bolt'}

        plot_start = time.perf_counter()
        annotated_frame = result.plot()
        plot_ms = (time.perf_counter() - plot_start) * 1000

        #Time OpenCV display
        display_start = time.perf_counter()
        cv2.imshow('Azure Kinect YOLO Segmentation', annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        display_ms = (time.perf_counter() - display_start) * 1000
        """
        self.get_logger().info(
            f"Plot: {plot_ms:.2f} ms | "
            f"Display: {display_ms:.2f} ms"
        )
        """



        completed_at = time.perf_counter()
        if self.last_frame_completed_at is not None:
            self.frame_intervals.append(completed_at - self.last_frame_completed_at)
        self.last_frame_completed_at = completed_at

        # Log once per second rather than every frame (less logging overhead).
        if completed_at - self.last_status_log_at >= 1.0:
            actual_fps = (
                len(self.frame_intervals) / sum(self.frame_intervals)
                if self.frame_intervals else 0.0
            )
            """
            self.get_logger().info(
                f'Predict: {predict_ms:.2f} ms '
                f'({1000.0 / predict_ms:.1f} equivalent FPS) | '
                f'Pre: {speed.get("preprocess", 0.0):.2f} ms | '
                f'Model: {speed.get("inference", 0.0):.2f} ms | '
                f'Post: {speed.get("postprocess", 0.0):.2f} ms | '
                f'Processed: {actual_fps:.1f} FPS (rolling)'
            )
            """
            self.last_status_log_at = completed_at

        if key == ord('q') and rclpy.ok():
            rclpy.shutdown()

    def print_benchmark_results(self):
        times = self.inference_times
        sorted_times = sorted(times)
        avg_ms = statistics.mean(times)
        p95_ms = sorted_times[int(0.95 * (len(sorted_times) - 1))]

        self.get_logger().info(
            '\n========== YOLO GPU/CPU BENCHMARK ==========\n'
            f'Device:             {self.device}\n'
            f'Measured frames:    {len(times)}\n'
            f'Average predict:   {avg_ms:.2f} ms\n'
            f'Std deviation:     {statistics.stdev(times):.2f} ms\n'
            f'Min predict:       {min(times):.2f} ms\n'
            f'Max predict:       {max(times):.2f} ms\n'
            f'P95 predict:       {p95_ms:.2f} ms\n'
            f'Equivalent FPS:    {1000.0 / avg_ms:.2f}\n'
            f'Avg preprocessing: {statistics.mean(self.preprocess_times):.2f} ms\n'
            f'Avg model:         {statistics.mean(self.model_times):.2f} ms\n'
            f'Avg postprocess:   {statistics.mean(self.postprocess_times):.2f} ms\n'
            '============================================'
        )

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KinectSegmentationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
