#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist

from cv_bridge import CvBridge
import cv2
import numpy as np


class VisualOdometry(Node):

    def __init__(self):
        super().__init__('visual_odometry')

        self.get_logger().info("VO node started")

        self.bridge = CvBridge()

        # Subscribers
        self.sub = self.create_subscription(
            Image,
            '/new_rover/oakd_rgb/image_color',
            self.image_callback,
            10)

        # Publisher
        self.pub = self.create_publisher(Odometry, '/visual_odom', 10)

        # ORB detector
        self.orb = cv2.ORB_create(2000)

        # Previous frame
        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None

        # Pose
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        # Camera intrinsic (⚠️ CHANGE THIS for your camera)
        self.K = np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

    # ─────────────────────────────────────────────
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_img is None:
            self.prev_img = gray
            self.prev_kp = kp
            self.prev_des = des
            return

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.prev_des, des)

        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            return

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        # Essential matrix
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K)

        if E is None:
            return

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        # Accumulate pose
        self.t += self.R @ t
        self.R = R @ self.R

        self.publish_odometry()

        # Update previous frame
        self.prev_img = gray
        self.prev_kp = kp
        self.prev_des = des

    # ─────────────────────────────────────────────
    def publish_odometry(self):

        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = float(self.t[0])
        msg.pose.pose.position.y = float(self.t[1])
        msg.pose.pose.position.z = float(self.t[2])

        # Convert rotation matrix to quaternion
        q = self.rotation_to_quaternion(self.R)

        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        self.pub.publish(msg)

    # ─────────────────────────────────────────────
    def rotation_to_quaternion(self, R):

        q = np.zeros(4)
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2, 1] - R[1, 2]) * s
            q[1] = (R[0, 2] - R[2, 0]) * s
            q[2] = (R[1, 0] - R[0, 1]) * s

        return q


def main():
    rclpy.init()
    node = VisualOdometry()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()