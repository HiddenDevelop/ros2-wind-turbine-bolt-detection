
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class KinectVideoRecorder(Node):
    def __init__(self, duration=10):
        super().__init__('kinect_video_recorder')

        self.subscription = self.create_subscription(
            Image,
            '/k4a/rgb/image_raw',
            self.image_callback,
            10
        )

        self.br = CvBridge()
        self.duration = duration  # Recording duration in seconds
        self.start_time = None
        self.video_writer = None
        self.recording_finished = False

        self.filename = 'ur10e_kinect_recorded_video.mp4'
        self.fps = 30.0

        self.get_logger().info(
            f'Waiting for Kinect frames to record {duration} seconds...'
        )

    def image_callback(self, msg):
        if self.recording_finished:
            return

        try:
            current_frame = self.br.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )

            # Initialize video writer on first frame
            if self.video_writer is None:
                height, width, _ = current_frame.shape

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.filename,
                    fourcc,
                    self.fps,
                    (width, height)
                )

                if not self.video_writer.isOpened():
                    raise RuntimeError('Failed to open video writer')

                self.start_time = time.monotonic()
                self.get_logger().info('Recording started!')

            # Stop after specified duration
            elapsed_time = time.monotonic() - self.start_time

            if elapsed_time >= self.duration:
                self.video_writer.release()
                self.video_writer = None
                self.recording_finished = True

                self.get_logger().info(
                    f'Successfully saved video to {self.filename}'
                )
                return

            # Save frame to video
            self.video_writer.write(current_frame)

        except Exception as e:
            self.get_logger().error(
                f'Failed to record video: {e}'
            )
            self.recording_finished = True

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None


def main(args=None):
    rclpy.init(args=args)

    # Change this to the desired recording duration
    recording_duration = 10  # seconds

    node = KinectVideoRecorder(duration=recording_duration)

    try:
        while rclpy.ok() and not node.recording_finished:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info('Recording interrupted.')

    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
