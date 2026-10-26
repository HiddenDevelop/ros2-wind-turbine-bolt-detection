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

        self.last_print_time = self.get_clock().now()
        
        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        # Synchronize RGB and depth
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

        self.get_logger().info('Kinect RGB + Depth subscriber started')

    # Added 'self' to class methods
    def auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)

    def get_internal_angles(self, approx):
        angles = []
        for i in range(6):
            p1 = approx[i][0]
            p2 = approx[(i+1) % 6][0]
            p0 = approx[(i-1) % 6][0]
            
            v1 = p0 - p1
            v2 = p2 - p1
            
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            angles.append(angle)
        return angles
    
    def camera_info_callback(self, msg):
        # Fixed indentation
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # Added required callback arguments
    def kinect_callback(self, rgb_msg, depth_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Limit depth values to a maximum of 60 cm and a min of 10 cm
            min_depth = 0.100
            max_depth = 0.600

            depth_valid = (
                np.isfinite(depth_image) &
                (depth_image >= min_depth) &
                (depth_image <= max_depth)
            )

            # ARCHITECTURE UPGRADE: 
            # Apply the depth mask to the RGB image. 
            # This turns everything further than 60cm or closer than 10cm pitch black,
            # effectively "isolating" the object like it was on the white piece of paper.
            mask = np.uint8(depth_valid) * 255
            isolated_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            
            # Use the isolated image instead of the messy full scene
            #gray = cv2.cvtColor(isolated_rgb, cv2.COLOR_BGR2GRAY) # Test 1
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) # Test 2


            enhanced_gray = clahe.apply(gray)
            blurred = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Call using self.
            edged = self.auto_canny(blurred)
                
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))         

            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)


            contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            hex_coordinates = []

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                if peri < 80: 
                    continue
                    
                #epsilon = 0.045 * peri # Test 1
                #epsilon = 0.025 * peri # Test 2
                epsilon = 0.02 * peri # Test 3

                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 6 and cv2.isContourConvex(approx):
                    # Call using self.
                    angles = self.get_internal_angles(approx)
                    
                    if min(angles) > 90 and max(angles) < 150:

                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                            
                        if 0.4 <= aspect_ratio <= 2.5:
                            cv2.drawContours(rgb_image, [approx], -1, (0, 255, 0), 3)
                                
                            #M = cv2.moments(cnt)

                            M = cv2.moments(approx)

                            if M["m00"] != 0:

                                # 2-D centroid
                                cX = int(M["m10"] / M["m00"])

                                cY = int(M["m01"] / M["m00"])

                                # Draw centroid
                                cv2.circle(
                                    rgb_image,
                                    (cX, cY),
                                    5,
                                    (0, 0, 255),
                                    -1
                                )

                                # -----------------------------------
                                # Convert centroid into camera XYZ
                                # -----------------------------------

                                if (
                                    self.fx is not None and
                                    self.fy is not None and
                                    self.cx is not None and
                                    self.cy is not None and
                                    self.fx != 0 and
                                    self.fy != 0
                                ):
                                    if (
                                        0 <= cX < depth_image.shape[1] and
                                        0 <= cY < depth_image.shape[0]
                                    ):
                                        z_cam = float(depth_image[cY, cX])

                                        if (np.isfinite(z_cam) and
                                            min_depth <= z_cam <= max_depth
                                        ):

                                            x_cam = (
                                                (cX - self.cx)* z_cam / self.fx
                                            )

                                            y_cam = ((cY - self.cy)* z_cam / self.fy)

                                            hex_coordinates.append(
                                                (cX, cY, x_cam, y_cam, z_cam)
                                            )

                                            # self.get_logger().info(
                                            #     f'Bolt center RGB frame: '
                                            #     f'X={x_cam:.4f} m, '
                                            #     f'Y={y_cam:.4f} m, '
                                            #     f'Z={z_cam:.4f} m'
                                            # )
                                            cv2.putText(
                                                rgb_image,
                                                f'({x_cam:.3f}, '
                                                f'{y_cam:.3f}, '
                                                f'{z_cam:.3f}) m',
                                                (cX - 60, cY + 25),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 0, 0),
                                                2
                                            )

                                    cv2.putText(
                                        rgb_image,
                                        "Hex Target",
                                        (cX - 40, cY - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2
                                    )


            current_time = self.get_clock().now()

            elapsed = (
                current_time - self.last_print_time
            ).nanoseconds / 1e9

            if elapsed >= 5.0:

                if len(hex_coordinates) > 0:

                    hex_coordinates.sort(
                        key=lambda h: h[0]
                    )

                    self.get_logger().info(
                        f'Detected {len(hex_coordinates)} hexagons'
                    )

                    for i, (cX, cY, x, y, z) in enumerate(
                        hex_coordinates,
                        start=1
                    ):
                        self.get_logger().info(
                            f'Hex {i}: '
                            f'pixel=({cX}, {cY}) | '
                            f'X={x:.4f} m, '
                            f'Y={y:.4f} m, '
                            f'Z={z:.4f} m'
                        )

                else:

                    self.get_logger().info(
                        'No valid hexagons detected.'
                    )

                self.last_print_time = current_time
            cv2.imshow("Robot Hex Localization", rgb_image)
            cv2.imshow("Edges", edged)
            cv2.imshow("Closed", closed)
            cv2.waitKey(1)
        
        except Exception as e:
            self.get_logger().error(f'Kinect processing error: {e}')

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