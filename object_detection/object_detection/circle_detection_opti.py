#!/usr/bin/env python



from visualization_msgs.msg import Marker 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()
        self.cloud_points=None
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        #self.depth_sub= self.create_subscription(PointCloud2, '/camera/camera/depth/color/points',self.cloud_callback,10) 
        self.marker_pub = self.create_publisher(Marker, '/detected_circles', 10)
        

     #def cloud_callback(self, msg):
     #   cloud_points=[]
     #   for point in msg.data:
     #       print(f"receive number {point}")
        """
        # Parse the PointCloud2 message to extract x, y, z coordinates
        self.cloud_points = []
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            self.cloud_points.append(point)
        self.get_logger().info(f"Received {len(self.cloud_points)} cloud points")
        """


    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            #the higher param2, the less circles we get , if we reduce it we get more false positive
            """
            Parameters 1 and 2 don't affect accuracy as such, more reliability. Param 1 will set 
            the sensitivity; how strong the edges of the circles need to be. Too high and it 
            won't detect anything, too low and it will find too much clutter. Param 2 will set how many 
            edge points it needs to find to declare that it's found a circle. Again, too high will detect 
            nothing, too low will declare anything to be a circle. The ideal value of param 2 will
            be related to the circumference of the circles.
            """
            circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                        param1=180, param2=38, minRadius=4, maxRadius=40)  # this parameters can be adjusted 
            #param1=100, param2=30, minRadius=10, maxRadius=30, dp=1.2, minDist=50,
                                       
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    coordinates_text = f"({x}, {y})"
                    cv2.putText(cv_image, coordinates_text, (x - 50, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv_image= cv2.circle(cv_image, (x, y), r, (0, 255, 0), 4)
            cv2.imshow("Circle Detection", cv_image)
            cv2.waitKey(1)
            #print(self.cloud_points)
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

