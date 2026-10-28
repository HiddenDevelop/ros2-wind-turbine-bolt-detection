import rclpy
import cv2
import numpy as np
import time

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

# Import YOLO (Assuming ultralytics YOLOv8/v11)
from ultralytics import YOLO
import math

def distance_formula(*coords) -> float:
    squared = 0
    for val in coords:
        squared += val**2
    return math.sqrt(squared)

class YoloDepthLocalization(Node):
    def __init__(self):
        super().__init__('yolo_depth_localization')

        self.bridge = CvBridge()
        
        # Load your custom trained YOLO model
        # Replace 'best.pt' with the path to your trained bolt model
        self.model = YOLO('best.pt') 

        #self.timer = self.create_timer(5.0, self.timer_callback)

        # Camera Intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None


        self.last_print_time = self.get_clock().now()

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        # Synchronize RGB and depth frames
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
        
        self.target_pub = self.create_publisher(Point, '/hex_bolt/spatial_coordinate', 10)

        self.get_logger().info('YOLO 3D Target Tracker started. Waiting for frames...')

    def timer_callback(self):
        self.get_logger().info()

    def camera_info_callback(self, msg):
        # Extract intrinsic parameters once
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(f'Intrinsics locked: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def kinect_callback(self, rgb_msg, depth_msg):
        if self.fx is None:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # 1. Run YOLO inference on the RGB frame
            results = self.model.predict(rgb_image, conf=0.3, verbose=False)
            
            positions = []

            # 2. Iterate through detected bounding boxes
            for index, box in enumerate(results[0].boxes):
                # Extract integer coordinates of the bounding box window
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure bounding box is within image dimensions to prevent slicing errors
                h, w = rgb_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 3. Calculate the global 2D centroid of the bounding box
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                # 4. Extract the matching "window" (Region of Interest) from the DEPTH array
                depth_window = depth_image[y1:y2, x1:x2]

                # Filter out invalid depth pixels (0.0 or infinite) within the window
                # Assuming kinect depth is in meters (if in mm, change 0.05 to 50, etc.)
                valid_depths = depth_window[(depth_window > 0.05) & np.isfinite(depth_window)]

                if len(valid_depths) > 0:
                    # 5. Use the MEDIAN depth of the window for robust 3D localization
                    # This ignores sensor noise, reflections, and gaps in the bolt
                    z_depth = float(np.median(valid_depths))
                    
                    # 6. Pinhole Camera Projection (2D pixel -> 3D coordinate)
                    point_x = (u - self.cx) * z_depth / self.fx
                    point_y = (v - self.cy) * z_depth / self.fy
                    point_z = z_depth
                    
                    # Publish the 3D spatial coordinate
                    spatial_msg = Point()
                    spatial_msg.x = point_x
                    spatial_msg.y = point_y
                    spatial_msg.z = point_z
                    self.target_pub.publish(spatial_msg)
                    positions.append( (point_x, point_y, point_z, index) )
                    
                    #self.get_logger().info(f"For {index}, the 3D coordinates are ({point_x}, {point_y}, {point_z})")
                                           # --- Draw UI Visualization ---
                    # Draw YOLO Bounding Box
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw Centroid Target
                    cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    
                    # Display the 3D Z depth on screen
                    depth_display = f"#Hex: {index} D: {distance_formula(point_x, point_y, point_z):.3f}m"
                    cv2.putText(rgb_image, depth_display, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            current_time = self.get_clock().now()

            elapsed = (
                current_time - self.last_print_time
            ).nanoseconds / 1e9

            if elapsed >= 2.0:
                self.get_logger().info(
                        "-------------------------------------------------------------------------------------"
                    )
                for position in positions:
                    self.get_logger().info(
                        f"For Hex #{position[3] + 1}, the 3D coordinates are ({position[0]}, {position[1]}, {position[2]})"
                    )


                self.last_print_time = current_time    

            cv2.imshow("YOLO 3D Target Localization", rgb_image)
            cv2.waitKey(1)
        
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()