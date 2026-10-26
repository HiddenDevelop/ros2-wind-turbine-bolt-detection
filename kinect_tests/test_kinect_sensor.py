import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge


class KinectSensor(Node):

    def __init__(self):
        super().__init__('kinect_sensor')

        # Converts ROS Image messages <-> OpenCV images
        self.bridge = CvBridge()

        self.declare_parameter(
            'rgb_topic',
            '/k4a/rgb/image_raw'
        )

        self.declare_parameter(
            'depth_topic',
            '/k4a/depth_to_rgb/image_raw'
        )

        self.declare_parameter(
            'point_cloud2_topic',
            '/k4a/points2'
        )

        rgb_topic = self.get_parameter(
            'rgb_topic'
        ).get_parameter_value().string_value

        depth_topic = self.get_parameter(
            'depth_topic'
        ).get_parameter_value().string_value

        point_cloud2_topic = self.get_parameter(
            'point_cloud2_topic'
        ).get_parameter_value().string_value

        self.rgb_sub = Subscriber(
            self,
            Image,
            rgb_topic
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            depth_topic
        )

        # Synchronize RGB and depth
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.kinect_callback)

        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            point_cloud2_topic,
            self.point_cloud_callback,
            10
        )

        # Used so we don't print thousands of messages
        self.cloud_count = 0

        self.get_logger().info(
            'Kinect RGB + Depth + PointCloud subscriber started'
        )


    def kinect_callback(self, rgb_msg, depth_msg):
        try:
            # Convert RGB to OpenCV image
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )

            # Convert depth to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding='passthrough'
            )

            self.get_logger().info(
                f'RGB: {rgb_image.shape} | '
                f'Depth: {depth_image.shape}'
            )

            # Display RGB
            cv2.imshow(
                "Kinect RGB",
                rgb_image
            )

            # Normalize depth for display
            depth_display = cv2.normalize(
                depth_image,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            depth_display = depth_display.astype(
                np.uint8
            )

            cv2.imshow(
                "Kinect Depth",
                depth_display
            )

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(
                f'Kinect processing error: {e}'
            )


    def point_cloud_callback(self, cloud_msg):
        try:
            self.cloud_count += 1

            # Read XYZ points from PointCloud2 message.
            # skip_nans=True removes invalid depth measurements.
            points = point_cloud2.read_points(
                cloud_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )

            # Convert generator / structured array into XYZ values
            xyz_points = []

            for point in points:
                # Handles common ROS2 PointCloud2 return formats
                x = float(point[0])
                y = float(point[1])
                z = float(point[2])

                xyz_points.append(
                    [x, y, z]
                )

            num_valid_points = len(xyz_points)

            # Print once every 30 point clouds so terminal
            # doesn't get flooded
            if self.cloud_count % 30 == 0:

                self.get_logger().info(
                    '---------------- POINT CLOUD TEST ----------------'
                )

                self.get_logger().info(
                    f'Frame: {cloud_msg.header.frame_id}'
                )

                self.get_logger().info(
                    f'Cloud size: '
                    f'{cloud_msg.width} x {cloud_msg.height}'
                )

                self.get_logger().info(
                    f'Valid XYZ points: {num_valid_points}'
                )

                self.get_logger().info(
                    f'Point step: {cloud_msg.point_step} bytes'
                )

                # Print first five valid XYZ coordinates
                if num_valid_points > 0:

                    self.get_logger().info(
                        'Sample points (meters):'
                    )

                    for i, point in enumerate(
                        xyz_points[:5]
                    ):
                        x, y, z = point

                        self.get_logger().info(
                            f'Point {i}: '
                            f'x={x:.3f}, '
                            f'y={y:.3f}, '
                            f'z={z:.3f}'
                        )

                    # Calculate useful statistics
                    xyz_array = np.asarray(
                        xyz_points,
                        dtype=np.float32
                    )

                    min_xyz = np.min(
                        xyz_array,
                        axis=0
                    )

                    max_xyz = np.max(
                        xyz_array,
                        axis=0
                    )

                    mean_xyz = np.mean(
                        xyz_array,
                        axis=0
                    )

                    self.get_logger().info(
                        f'X range: '
                        f'{min_xyz[0]:.3f} to '
                        f'{max_xyz[0]:.3f} m'
                    )

                    self.get_logger().info(
                        f'Y range: '
                        f'{min_xyz[1]:.3f} to '
                        f'{max_xyz[1]:.3f} m'
                    )

                    self.get_logger().info(
                        f'Z range: '
                        f'{min_xyz[2]:.3f} to '
                        f'{max_xyz[2]:.3f} m'
                    )

                    self.get_logger().info(
                        f'Mean Z distance: '
                        f'{mean_xyz[2]:.3f} m'
                    )

                    self.get_logger().info(
                        'POINT CLOUD GENERATION: WORKING'
                    )

                else:
                    self.get_logger().warn(
                        'PointCloud2 message received, '
                        'but it contains no valid XYZ points.'
                    )

        except Exception as e:
            self.get_logger().error(
                f'Point cloud processing error: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = KinectSensor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
