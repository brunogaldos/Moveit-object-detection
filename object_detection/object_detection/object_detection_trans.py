#!/usr/bin/env python
import rclpy
import csv
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
from realsense2_camera_msgs.msg import Metadata  # Import Metadata message type

class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()
        self.cloud_points = None
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        self.metadata_sub = self.create_subscription(Metadata, '/camera/camera/depth/metadata', self.metadata_callback, 10)  # Update subscription to Metadata
        self.depth_image = None
        self.camera_info = None
        self.frame_timestamp = None

        # Open CSV file for writing
        self.csv_file = open('circle_coordinates.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['x', 'y', 'z', 'frame_timestamp'])

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def metadata_callback(self, msg):
        self.frame_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # Update to handle Metadata message

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                       param1=180, param2=36, minRadius=4, maxRadius=40)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    coordinates_text = f"({x}, {y})"
                    cv_image = cv2.circle(cv_image, (x, y), r, (0, 255, 0), 4)

                    if self.depth_image is not None and self.camera_info is not None:
                        depth = self.depth_image[y, x] / 1000.0  # Convert depth to meters
                        camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
                        fx = camera_matrix[0, 0]
                        fy = camera_matrix[1, 1]
                        cx = camera_matrix[0, 2]
                        cy = camera_matrix[1, 2]

                        # Convert pixel coordinates to camera coordinates
                        X = (x - cx) * depth / fx
                        Y = (y - cy) * depth / fy
                        Z = depth
                        print(f"x:{X},y:{Y},z:{Z},timestamp:{self.frame_timestamp}")

                        # Write to CSV file
                        #self.csv_writer.writerow([X, Y, Z, self.frame_timestamp])

            cv2.imshow("Circle Detection", cv_image)
            cv2.waitKey(3)
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def __del__(self):
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    circle_detector = CircleDetector()
    rclpy.spin(circle_detector)
    circle_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()