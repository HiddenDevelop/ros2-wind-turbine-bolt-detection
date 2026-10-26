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

        rgb_topic = self.get_parameter(
            'rgb_topic'
        ).get_parameter_value().string_value

        depth_topic = self.get_parameter(
            'depth_topic'
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

        # Used so we don't print thousands of messages
        self.cloud_count = 0

        self.get_logger().info(
            'Kinect RGB + Depth subscriber started'
        )


    def auto_canny(image, sigma=0.33):
        """Dynamically calculates Canny thresholds based on lighting."""
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)

    def get_internal_angles(approx):
        """Calculates the internal angles of a polygon to verify its geometric shape."""
        angles = []
        for i in range(6):
            # Get 3 consecutive points
            p1 = approx[i][0]
            p2 = approx[(i+1) % 6][0]
            p0 = approx[(i-1) % 6][0]
            
            # Create vectors
            v1 = p0 - p1
            v2 = p2 - p1
            
            # Calculate angle between vectors using dot product
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            angles.append(angle)
        return angles
    
    def camera_info_callback(self, msg):
    
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def kinect_callback():

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

            # High-intensity CLAHE to separate dark objects from dark backgrounds
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                
            # Contrast Enhancement
            enhanced_gray = clahe.apply(gray)

            # Apply blur to remove surface texture but keep edges
            blurred = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)
                
            # Dynamic Edge Detection
            edged = auto_canny(blurred)
                
            # Morphological Closing
            # Aggressively dilates the edges 
            # to connect broken lines caused by the glare, then erodes back to normal size.

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                    
                # Micro-noise rejection to keep robotic framerates high
                if peri < 80: 
                    continue
                    
                # Eosilon Approximation
                epsilon = 0.045 * peri
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Check for exactly 6 sides and convexity
                if len(approx) == 6 and cv2.isContourConvex(approx):
                        
                    # Angle Verification
                    # A regular hexagon has 120-degree angles. Under 3D perspective skew, 
                    # these will stretch. Reject anything with highly acute angles (sharp spikes).

                    angles = get_internal_angles(approx)
                    if min(angles) > 70 and max(angles) < 170:
                            
                            # 8. Aspect Ratio Check (Scale-invariant)
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                            
                        if 0.4 <= aspect_ratio <= 2.5:
                                
                            # Draw detection outline
                            cv2.drawContours(rgb_image, [approx], -1, (0, 255, 0), 3)
                                
                            # Target Extraction for Depth Sensor
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                    
                                cv2.circle(rgb_image, (cX, cY), 5, (0, 0, 255), -1)
                                cv2.putText(rgb_image, "Hex Target", (cX - 40, cY - 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Robot Hex Localization", rgb_image)
                # Uncomment below to see the morphological closing bridge gaps in real-time
                # cv2.imshow("Edge Debug", closed) 

            cv2.waitKey(1)
        
        except Exception as e:
            self.get_logger().error(
                f'Kinect processing error: {e}'
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