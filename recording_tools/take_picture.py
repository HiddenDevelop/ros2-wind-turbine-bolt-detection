#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class KinectPictureTaker(Node):
    def __init__(self):
        super().__init__('kinect_picture_taker')
        
        # Using the exact topic name from your list
        self.subscription = self.create_subscription(
            Image,
            '/k4a/rgb/image_raw',
            self.image_callback,
            10
        )
        self.br = CvBridge()
        self.image_saved = False

    def image_callback(self, msg):
        if self.image_saved:
            return

        try:
            current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = 'ur10e_kinect_captured_image.png'
            cv2.imwrite(filename, current_frame)
            
            self.get_logger().info(f'Successfully saved picture to {filename}')
            self.image_saved = True
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert and save image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = KinectPictureTaker()
    
    while rclpy.ok() and not node.image_saved:
        rclpy.spin_once(node, timeout_sec=0.1)
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()