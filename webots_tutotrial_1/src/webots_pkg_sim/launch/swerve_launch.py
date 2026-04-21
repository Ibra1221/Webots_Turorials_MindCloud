#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge
import cv2
import numpy as np

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class VisualOdometry(Node):

    def __init__(self):
        super().__init__('visual_odometry')

        self.get_logger().info("VO node started")

        self.bridge = CvBridge()

        # Subscriber
        self.sub = self.create_subscription(
            Image,
            '/new_rover/oakd_rgb/image_color',
            self.image_callback,
            10)

        # Publisher
        self.pub = self.create_publisher(Odometry, '/visual_odom', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # ORB detector
        self.orb = cv2.ORB_create(1500)

        # Previous frame
        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None

        # Pose
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        # ⚠️ Replace with real camera intrinsics
        self.K = np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

    # ─────────────────────────────────────────────
    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 10:
            return

        if self.prev_img is None:
            self.prev_img = gray
            self.prev_kp = kp
            self.prev_des = des
            return

        # --- Feature Matching (ratio test) ---
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(self.prev_des, des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 10:
            return

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])

        # --- Essential Matrix ---
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        # --- Reject noisy jumps ---
        if np.linalg.norm(t) > 1.0:
            return

        # --- Apply scale (TUNE THIS!) ---
        scale = 0.05
        self.t += self.R @ (t * scale)
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
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(self.t[0])
        msg.pose.pose.position.y = float(self.t[1])
        msg.pose.pose.position.z = float(self.t[2])

        q = self.rotation_to_quaternion(self.R)

        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        self.pub.publish(msg)

        # --- TF BROADCAST ---
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

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
        else:
            q[3] = 1.0

        return q


def main():
    rclpy.init()
    node = VisualOdometry()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()