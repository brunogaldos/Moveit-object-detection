import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/home/athena/Pictures/Screenshots/object_from_up.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply GaussianBlur to reduce noise and improve detection
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using HoughCircles
#circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
#                           param1=100, param2=30, minRadius=10, maxRadius=30)

#param1 is how strong edges of circle has to be, low_detect all, too_high detect nothing only strong edge circles
# Param 2 will set how many edge points it needs to find to declare that it's found a circle. Again, too high will detect nothing, too low will declare anything to be a circle
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                            param1=100, param2=28, minRadius=2, maxRadius=30)
# Draw detected circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (255, 0, 0), 4)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Circles')
plt.show()

