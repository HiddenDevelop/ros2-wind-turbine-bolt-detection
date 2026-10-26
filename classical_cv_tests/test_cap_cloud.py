import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge


class KinectSensor(Node):

    def __init__(self):
        super().__init__('kinect_sensor')

        # Converts ROS Image messages <-> OpenCV images
        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

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

        # self.point_cloud_sub = self.create_subscription(
        #     PointCloud2,
        #     point_cloud2_topic,
        #     self.point_cloud_callback,
        #     10
        # )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/k4a/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        self.object_cloud_pub = self.create_publisher(
            PointCloud2,
            '/object_points',
            10
        )

        # Used so we don't print thousands of messages
        self.cloud_count = 0

        self.get_logger().info(
            'Kinect RGB + Depth + PointCloud subscriber started'
        )

    def camera_info_callback(self, msg):
    
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]


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
            
            #Limit depth values to a maximum of 70 cm and a min of 30 cm (for testing purposes)
            min_depth = 0.100
            max_depth = 0.600

            depth_valid = (
                np.isfinite(depth_image) &
                (depth_image >= min_depth) &
                (depth_image <= max_depth)
            )
            # Apply HSV mask to filter out background except for bolt cap
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            low_bound = np.array([0, 0, 0], dtype=np.uint8)
            high_bound = np.array([179, 50, 50], dtype=np.uint8) #specifying shades of black for object - changed (n, 255, 70) -> (n, 50, 50)

            mask = cv2.inRange(hsv, low_bound, high_bound)

            color_valid = mask > 0

            combined_mask = depth_valid & color_valid

            filtered_depth = np.zeros_like(depth_image)
            filtered_depth[combined_mask] = depth_image[combined_mask]
            
            filtered_rgb = cv2.bitwise_and(
                rgb_image,
                rgb_image,
                mask=(combined_mask.astype(np.uint8) * 255)
            )
            
            # Verify RGB and aligned depth have matching image dimensions
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                self.get_logger().error(
                    f'RGB/depth size mismatch: '
                    f'{rgb_image.shape[:2]} vs {depth_image.shape[:2]}'
                )
                return

            if (
                self.fx is not None and
                self.fy is not None and
                self.fx != 0 and
                self.fy != 0
            ):

                v_coords, u_coords = np.where(combined_mask)

                if len(u_coords) > 0:

                    # 32FC1 depth is already in meters
                    z = depth_image[
                        v_coords,
                        u_coords
                    ].astype(np.float32)

                    x = (
                        (u_coords.astype(np.float32) - self.cx)
                        * z / self.fx
                    )

                    y = (
                        (v_coords.astype(np.float32) - self.cy)
                        * z / self.fy
                    )

                    xyz_points = np.column_stack(
                        (x, y, z)
                    ).astype(np.float32)

                else:
                    # No bolt detected this frame
                    xyz_points = np.empty(
                        (0, 3),
                        dtype=np.float32
                    )

                # Always publish, even when empty
                object_cloud_msg = point_cloud2.create_cloud_xyz32(
                    rgb_msg.header,
                    xyz_points.tolist()
                )

                self.object_cloud_pub.publish(
                    object_cloud_msg
                )

                self.get_logger().info(
                    f'Object Cloud Points: {len(xyz_points)}'
                )
                    
            self.get_logger().info(
                f'RGB: {rgb_image.shape} | '
                f'Depth: {depth_image.shape} | '
                f'Depth encoding: {depth_msg.encoding}'
            )

            # Display RGB
            cv2.imshow(
                "Kinect RGB",
                rgb_image
            )
            
            # Display Filtered RGB Image
            cv2.imshow(
                "Filtered Kinect RGB",
                filtered_rgb
            )

            #Display the binary color mask
            cv2.imshow(
                "Combined Mask",
                combined_mask.astype(np.uint8) * 255
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
            
            # Normalize filtered depth for display
            filtered_depth_display = cv2.normalize(
                filtered_depth,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            filtered_depth_display = filtered_depth_display.astype(
                np.uint8
            )

            # show depth image
            cv2.imshow(
                "Kinect Depth",
                depth_display
            )
            
            # show the filtered depth image
            cv2.imshow(
                "Filtered Kinect Depth",
                filtered_depth_display
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

            min_z = 0.100
            max_z = 0.600

            xyz_points = []

            for point in points:
                # Handles common ROS2 PointCloud2 return formats
                x = float(point[0])
                y = float(point[1])
                z = float(point[2])

                if min_z <= z <= max_z:
                    xyz_points.append(
                        [x, y, z]
                    )

            # xyz_points.append(
            #     [x, y, z]
            #         )

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
