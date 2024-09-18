
# Panda Robot Arm Picking Task with RealSense D435 Camera and Moveit2

## Overview

This project demonstrates a complete picking task using a Panda robot arm equipped with an Intel RealSense D435 camera. The objective is to detect a cylindrical object using an object detection script, compute the object's coordinates through frame transformation, and subsequently control the Panda arm to grasp and place the object at a designated goal pose. The project uses ROS2 and MoveIt for handling robot motion and perception.

### Key Features
- **Object Detection**: Detects the circular surface of a cylindrical object using a RealSense D435 camera.
- **Coordinate Frame Transformation**: Converts object coordinates from camera frame to robot coordinate system.
- **Grasp and Place**: Executes a pick-and-place task using the Panda robot arm controlled via MoveIt.
  
---

## Hardware Requirements
- **Franka Emika Panda Robot Arm**
- **Intel RealSense D435 Camera**

---

## Software Requirements
- ROS2 (tested on Humble)
- MoveIt2
- RealSense ROS2 Wrapper for D435 camera
- Custom Object detection package

---

## Setup Instructions

1. **Install ROS2**  
   Follow the official [ROS2 installation guide](https://docs.ros.org/en/foxy/Installation.html) for your operating system.

2. **Install MoveIt2**  
   Install MoveIt2 by following the official [MoveIt2 installation guide](https://moveit.ros.org/install-moveit2.html).

3. **Install RealSense ROS2 Wrapper**  
   To interface with the Intel RealSense D435 camera, you need to install the ROS2 wrapper:
   ```bash
   sudo apt-get install ros-<ros2-distro>-realsense2-camera
   ```

4. **Clone the Project Repository**  
   Clone this repository to your workspace:
   ```bash
   git clone https://github.com/brunogaldos/Moveit-object-detection.git
   ```

5. **Build the Workspace**
   After cloning the repository, build the workspace:
   ```bash
   cd ~/Moveit_object_detection
   colcon build
   ```

---

## Running the System

### 1. Start the RealSense D435 Camera
   Launch the RealSense camera to capture RGB-D images for object detection:
   ```bash
   ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
   ```

### 2. Run Object Detection Script
   Execute the object detection node to detect the circular base of the cylinder and output its position in the robot's coordinate frame:
   ```bash
   ros2 run object_detection object_detection_transformed
   ```

### 3. MoveIt Pick and Place Task
   Launch the MoveIt pipeline to perform the pick-and-place task using the Panda robot arm:
   ```bash
   ros2 launch moveit2_tutorials mtc_demo.launch.py
   ros2 launch mtc_tutorial pick_place_demo.launch.py
   ```

---

## Project Workflow

1. **Object Detection**  
   The RealSense D435 camera is used to capture RGB-D data, and the object detection script identifies the circular surface of the cylindrical object in the scene. The script outputs the detected object's pose in the camera frame.
![Screenshot from 2024-09-18 12-09-35](https://github.com/user-attachments/assets/13014819-d283-439b-9388-cbee4c51d01d)


2. **Coordinate Transformation**  
   The detected object's pose is transformed from the camera's coordinate frame to the robot's coordinate frame using appropriate TF (transform) methods.
![panda_arm_camera](https://github.com/user-attachments/assets/89d97452-a852-4cef-a215-00d579b9a558)

3. **Pick-and-Place Task**  
   The transformed coordinates are fed into the MoveIt pipeline, which generates a motion plan for the Panda robot arm. The arm grasps the cylinder and places it in the designated goal pose.
![Pick-and-Place Task](https://github.com/user-attachments/assets/ae455c09-df8d-48a1-978c-c6304873cc88)

---

## Future Improvements
- Improve object detection for non-circular or irregular objects.
- Add support for dynamic objects and real-time tracking.
- Implement feedback control to improve grasp stability.

---

## Acknowledgments
Special thanks to the ROS2, MoveIt, and RealSense communities for their invaluable open-source contributions.

---
