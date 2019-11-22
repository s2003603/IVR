#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # initialize a publisher to send joints' angular position to a topic called joints_pos
    self.joints_pub = rospy.Publisher("joints_pos",Float64MultiArray, queue_size=10)
    # initialize a publisher to send robot end-effector position
    self.end_effector_pub = rospy.Publisher("end_effector_prediction",Float64MultiArray, queue_size=10)
    # initialize a publisher to send desired trajectory
    self.trajectory_pub = rospy.Publisher("trajectory",Float64MultiArray, queue_size=10)
    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    # record the begining time
    self.time_trajectory = rospy.get_time()
    # initialize errors
    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')     
    self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')   
    # initialize error and derivative of error for trajectory tracking  
    self.error = np.array([0.0,0.0], dtype='float64')  
    self.error_d = np.array([0.0,0.0], dtype='float64') 

  def detect_red(self, image):
    mask = cv2.inRange(image, (0,0,100), (0,0,255))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    try:
      cx = int(M['m10'] / M['m00'])
    except:
      cx = 0
    try:
      cy = int(M['m01'] / M['m00'])
    except:
      cy = 0
    return np.array([cx, cy])

  def detect_green(self, image):
    mask = cv2.inRange(image, (0,100,0), (0,255,0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    try:
      cx = int(M['m10'] / M['m00'])
    except:
      cx = 0
    try:
      cy = int(M['m01'] / M['m00'])
    except:
      cy = 0
    return np.array([cx, cy])

  def detect_blue(self, image):
    mask = cv2.inRange(image, (100,0,0), (255,0,0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    try:
      cx = int(M['m10'] / M['m00'])
    except:
      cx = 0
    try:
      cy = int(M['m01'] / M['m00'])
    except:
      cy = 0
    return np.array([cx, cy])

  def detect_yellow(self, image):
    mask = cv2.inRange(image, (0,100,100), (0,255,255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    M = cv2.moments(mask)
    try:
      cx = int(M['m10'] / M['m00'])
    except:
      cx = 0
    try:
      cy = int(M['m01'] / M['m00'])
    except:
      cy = 0
    return np.array([cx, cy])

  def pixel2meter(self, image):
    circle1Pos = self.detect_blue(image)
    circle2Pos = self.detect_green(image)
    dist = np.sum((circle1Pos - circle2Pos)**2)
    return 3/np.sqrt(dist)

  def detect_joint_angles(self, image):
    a = self.pixel2meter(image)
    center = a*self.detect_yellow(image)
    circle1Pos = self.detect_blue(image)
    circle2Pos = self.detect_green(image)
    circle3Pos = self.detect_red(image)
    ja1 = np.arctan2(center[0] - circle1Pos[0], center[1] - circle1Pos[1])
    ja2 = np.arctan2(circle1Pos[0] - circle2Pos[0], circle1Pos[1] - circle2Pos[1]) - ja1
    ja3 = np.arctan2(circle2Pos[0] - circle3Pos[0], circle2Pos[1] - circle3Pos[1]) - ja2 - ja1
    return np.array([ja1, ja2, ja3])

  def detect_targets(self, image):
    mask = cv2.inRange(image, (50,120,120), (100,200,255))
    _, binarized = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 1:
      #ele's are superimposed from this angle, can only happen when they are straight on, so 
      #x coord is aligned with base (unchanging x-coord), and we get other val's from yz plane.
      x = self.detect_yellow(image)[0]
      return np.array([x, np.nan])
    else:
      #we don't need to use any advanced shape matching, just look at contours area and compare
      #to minimal bounding rectangle area; since shapes are around same size, we can just take
      #whichever has minimal error
      error1 = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(cnts[0]))) - cv2.contourArea(cnts[0])
      error2 = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(cnts[1]))) - cv2.contourArea(cnts[1])
      if error1 <= error2:
        cx = cv2.moments(cnts[0])['m10']/cv2.moments(cnts[0])['m00']
        cy = cv2.moments(cnts[0])['m01']/cv2.moments(cnts[0])['m00']
        return np.array([cx, cy])
      else:
        cx = cv2.moments(cnts[1])['m10']/cv2.moments(cnts[1])['m00']
        cy = cv2.moments(cnts[1])['m01']/cv2.moments(cnts[1])['m00']
        return np.array([cx, cy])

  def detect_end_effector(self, image):
    a = self.pixel2meter(image)
    endPos = a*(self.detect_yellow(image) - self.detect_red(image))
    return endPos

  def forward_kin(self, image):
    joints = self.detect_joint_angles(image)
    end_effector = np.array([3*np.sin(joints[0]) + 3*np.sin(joints[0] + joints[1]) + 3*np.sin(joints.sum()), 3*np.cos(joints[0]) + 3*np.cos(joints[0] + joints[1]) + 3*np.cos(joints.sum())])
    return end_effector

  def calculate_jacobian(self,image):
    joints = self.detect_joint_angles(image)
    jacobian = np.array([[3 * np.cos(joints[0]) + 3 * np.cos(joints[0]+joints[1]) + 3 *np.cos(joints.sum()), 3 * np.cos(joints[0]+joints[1]) + 3 *np.cos(joints.sum()),  3 *np.cos(joints.sum())], [-3 * np.sin(joints[0]) - 3 * np.sin(joints[0]+joints[1]) - 3 * np.sin(joints.sum()), - 3 * np.sin(joints[0]+joints[1]) - 3 * np.sin(joints.sum()), - 3 * np.sin(joints.sum())]])
    return jacobian

  def trajectory(self, image):
    return self.detect_targets(image)

  def control_closed(self,image):
    # P gain
    K_p = np.array([[10,0],[0,10]])
    # D gain
    K_d = np.array([[0.1,0],[0,0.1]])
    # estimate time step
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time
    # robot end-effector position
    pos = self.detect_end_effector(image)
    # desired trajectory
    pos_d= self.trajectory(image) 
    # estimate derivative of error
    self.error_d = ((pos_d - pos) - self.error)/dt
    # estimate error
    self.error = pos_d-pos
    q = self.detect_joint_angles(image) # estimate initial value of joints'
    J_inv = np.linalg.pinv(self.calculate_jacobian(image))  # calculating the psudeo inverse of Jacobian
    dq_d =np.dot(J_inv, ( np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose()) ) )  # control input (angular velocity of joints)
    q_d = q + (dt * dq_d)  # control input (angular position of joints)
    return q_d

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Perform image processing task (your code goes here)
    # The image is loaded as cv_imag

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    cv2.imshow('window', cv_image1)
    cv2.waitKey(3)
    
    # publish robot joints angles (lab 1 and 2)
    self.joints=Float64MultiArray()
    self.joints.data= self.detect_joint_angles(cv_image1)

    
    # compare the estimated position of robot end-effector calculated from images with forward kinematics(lab 3)
    x_e = self.forward_kin(cv_image1)
    x_e_image = self.detect_end_effector(cv_image1)
    self.end_effector=Float64MultiArray()
    self.end_effector.data= x_e_image	

    # send control commands to joints (lab 3)
    q_d = self.control_closed(cv_image1)
    self.joint1=Float64()
    self.joint1.data= q_d[0]
    self.joint2=Float64()
    self.joint2.data= q_d[1]
    self.joint3=Float64()
    self.joint3.data= q_d[2]

    # Publishing the desired trajectory on a topic named trajectory(lab 3)
    x_d = self.trajectory(cv_image1)    # getting the desired trajectory
    self.trajectory_desired= Float64MultiArray()
    self.trajectory_desired.data=x_d

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(cv_image1, "bgr8"))
      self.joints_pub.publish(self.joints)
      self.end_effector_pub.publish(self.end_effector)
      self.trajectory_pub.publish(self.trajectory_desired)
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)



