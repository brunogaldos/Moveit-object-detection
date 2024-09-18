#!/usr/bin/env python




from visualization_msgs.msg import Marker 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.detected_circles = []  # Using a list to store unique circles
        self.marker_pub = self.create_publisher(Marker, '/detected_circles', 10)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=100,
                                        param1=200, param2=24.5, minRadius=13, maxRadius=26)  # this parameters can be adjusted 

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    circle_roi = gray[y-r:y+r, x-r:x+r]
                    
                    if circle_roi.shape[0] > 0 and circle_roi.shape[1] > 0:
                        edges = cv2.Canny(circle_roi, 100, 200)
                        if np.mean(edges) > 45:
                            self.add_to_detected_circles((x, y, r))
            
            for (x, y, r) in self.detected_circles:
                cv_image = self.draw_frame(cv_image, x, y, r)
                cv_image = cv2.circle(cv_image, (x, y), r, (0, 255, 0), 4)
                coordinates_text = f"({x}, {y})"
                cv2.putText(cv_image, coordinates_text, (x - 50, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                self.publish_marker(float(x), float(y), float(r))   
            cv2.imshow("Circle Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    


    def publish_marker(self, x, y, r):
        marker = Marker()
        marker.header.frame_id = "camera_link"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = r * 2.0
        marker.scale.y = r * 2.0
        marker.scale.z = r * 2.0
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0  # Assuming circles are detected in 2D image plane

        self.marker_pub.publish(marker)




    def add_to_detected_circles(self, new_circle):
        tolerance = 50  # Define tolerance for considering circles as the same


        for (x, y, r) in self.detected_circles:
            if abs(x - new_circle[0]) < tolerance and abs(y - new_circle[1]) < tolerance:
                return  # Circle is considered already detected

        self.detected_circles.append(new_circle)

    def draw_frame(self, image, x, y, r):
        arrow_length = r * 2.0
        fx = 970.089
        fy = 970.089
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        points_3d = np.float32([[0, 0, 0], [arrow_length, 0, 0], [0, arrow_length, 0], [0, 0, arrow_length]]).reshape(-1, 3)
        tvec = np.array([x - cx, y - cy, 0], dtype=np.float32).reshape(3, 1)
        rvec = np.zeros((3, 1), dtype=np.float32)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        z_axis_end = (x + int(points_2d[3][0][0]), y + int(points_2d[3][0][1]))
        cv2.line(image, (x, y), z_axis_end, (255, 0, 0), 2)
        return image

def main(args=None):
    rclpy.init(args=args)
    circle_detector = CircleDetector()
    try:
        rclpy.spin(circle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        circle_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

