#! /usr/bin/env python3
"""
Visual Odometry node for the OAK-D camera in Webots/ROS2 Jazzy.
Tracks base_link pose in the odom frame using RGB + depth images.
"""

import os
import rclpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


class OAKVisualOdometry(Node):

    def __init__(self):
        super().__init__("oak_visual_odom")

        # ── Publishers ─────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)

        # ── CV tools ───────────────────────────────────────────────
        self.bridge        = CvBridge()
        self.orb           = cv2.ORB.create(3000)
        self.index_params  = dict(algorithm=6, trees=5)
        self.search_params = dict(checks=50)

        # ── State ──────────────────────────────────────────────────
        self.K              = None
        self.last_timestamp = None
        self.prev_rgb_img   = None
        self.prev_depth_img = None
        self.estimates      = []

        # ── Build optical → base_link transform ────────────────────
        #
        # WHY: solvePnP returns pose in the depth optical frame.
        #      Odometry must be reported with base_link as child frame.
        #      We build the full chain once here and reuse every frame.
        #
        # Step A: optical → oakd_link
        #   Static TF from driver: quat (x=-0.5, y=0.5, z=-0.5, w=0.5)
        #   Standard ROS optical convention rotation.
        # NEW — use exact matrix from tf2_echo instead of guessing quaternions
        # tf2_echo base_link oakd_depth_optical_link reported:
        #   Translation: [0.400, 0.000, 0.150]
        #   Matrix:
        #     0.000  0.000  1.000  0.400
        #    -1.000  0.000  0.000  0.000
        #     0.000 -1.000  0.000  0.150
        #     0.000  0.000  0.000  1.000
        T_base_to_optical = np.array([
            [ 0.0,  0.0,  1.0,  0.4 ],
            [-1.0,  0.0,  0.0,  0.0 ],
            [ 0.0, -1.0,  0.0,  0.15],
            [ 0.0,  0.0,  0.0,  1.0 ]
        ])

        # Invert analytically — more stable than np.linalg.inv for rigid transforms
        # R_inv = R^T,  t_inv = -R^T * t
        R_b2o = T_base_to_optical[:3, :3]
        t_b2o = T_base_to_optical[:3,  3]

        self.T_optical_to_base         = np.eye(4)
        self.T_optical_to_base[:3, :3] = R_b2o.T
        self.T_optical_to_base[:3,  3] = -(R_b2o.T @ t_b2o)

        self.T_base_to_optical = T_base_to_optical

        # Sanity check — logs Identity if the inversion is correct
        check = self.T_base_to_optical @ self.T_optical_to_base
        self.get_logger().info(
            f"Transform sanity check (expect Identity):\n"
            f"{np.round(check, 4)}")

        # C_k starts as T_base_to_optical so that at t=0:
        # C_k @ T_optical_to_base = T_base_to_optical @ T_optical_to_base = I
        # meaning base_link begins exactly at the odom origin
        self.C_k = self.T_base_to_optical.copy()
        # ── Subscribe to camera info first ─────────────────────────
        self.calib_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            10
        )

        self.get_logger().info("OAK Visual Odometry node initialized")

    # ──────────────────────────────────────────────────────────────
    def camera_info_callback(self, msg):
        """Grab camera intrinsics once, then set up image subscribers."""
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("Camera calibration received")
            self.setup_synchronized_subscribers()
            self.destroy_subscription(self.calib_sub)

    # ──────────────────────────────────────────────────────────────
    def setup_synchronized_subscribers(self):
        """Set up time-synchronized RGB + depth subscribers."""
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        self.get_logger().info("Synchronized subscribers set up")

    # ──────────────────────────────────────────────────────────────
    def synchronized_callback(self, rgb_msg, depth_msg):
        """Decode images and forward to frame processor."""
        self.get_logger().info("In callback")
        try:
            # WHY bgr8 → grayscale:
            #   Driver publishes bgra8 (Webots native).
            #   Requesting "mono8" directly from CvBridge with a bgra8
            #   source produces garbage pixels → bad ORB keypoints.
            #   bgr8 lets CvBridge drop the alpha channel cleanly.
            bgr_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            # WHY explicit 32FC1:
            #   Driver publishes depth as float32 meters (32FC1).
            #   Being explicit prevents silent failure if the encoding
            #   field is ever wrong, and guarantees scale_factor=1.0.
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="32FC1")

            if (self.prev_rgb_img   is not None and
                    self.prev_depth_img is not None):
                self.process_frames(
                    self.prev_rgb_img, rgb_img,
                    self.prev_depth_img, depth_img,
                    rgb_msg.header.stamp
                )

            self.prev_rgb_img   = rgb_img
            self.prev_depth_img = depth_img
            self.last_timestamp = rgb_msg.header.stamp

        except Exception as e:
            self.get_logger().error(
                f"Error in synchronized callback: {e}")

    # ──────────────────────────────────────────────────────────────
    def process_frames(self, prev_rgb, curr_rgb,
                       prev_depth, curr_depth, timestamp):
        """Core VO pipeline: match → 3D → PnP → accumulate → publish."""
        try:
            # ── Feature matching ───────────────────────────────────
            pts1, pts2, _ = self.match_features(prev_rgb, curr_rgb)
            if pts1 is None or len(pts1) < 8:
                self.get_logger().warn("Not enough good matches")
                return

            # ── Lift 2D → 3D using previous depth ─────────────────
            pts3d, valid_idx = self.get_3d_points(pts1, prev_depth)
            if pts3d is None or len(pts3d) < 4:
                self.get_logger().warn("Not enough valid 3D points")
                return

            pts2_valid = pts2[valid_idx]

            # ── Solve PnP ──────────────────────────────────────────
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2_valid, self.K, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success or inliers is None or len(inliers) < 5:
                self.get_logger().warn(
                    f"PnP failed, inliers: "
                    f"{0 if inliers is None else len(inliers)}")
                return

            # ── Sanity check ───────────────────────────────────────
            # WHY: One bad PnP result permanently corrupts C_k because
            #      poses accumulate. A rover cannot move >1m per frame
            #      at normal camera rates, so we reject such outliers.
            if np.linalg.norm(tvec) > 1.0:
                self.get_logger().warn(
                    f"Rejecting PnP: tvec magnitude "
                    f"{np.linalg.norm(tvec):.3f}m too large")
                return

            # ── Invert PnP transform ───────────────────────────────
            # WHY: solvePnP returns T such that p_cam = R*p_world + t,
            #      i.e. world→camera. We need camera motion in world
            #      (camera→world), so we invert:
            #        R_inv = R^T
            #        t_inv = -R^T * t
            R_mat, _ = cv2.Rodrigues(rvec)
            T_k_inv         = np.eye(4)
            T_k_inv[:3, :3] = R_mat.T
            T_k_inv[:3,  3] = -(R_mat.T @ tvec.flatten())

            # ── Accumulate optical-frame pose ──────────────────────
            self.C_k = self.C_k @ T_k_inv

            # ── Express base_link pose in odom frame ───────────────
            # WHY: C_k tracks the optical frame in odom.
            #      Multiplying on the right by T_optical_to_base gives
            #      us where base_link is in odom.
            #      Order matters: accumulated motion FIRST, then offset.
            #        C_k @ T_optical_to_base  ← CORRECT
            #        T_optical_to_base @ C_k  ← WRONG (wrong order)
            C_k_robot = self.C_k @ self.T_optical_to_base

            # ── Publish ────────────────────────────────────────────
            odom_msg = self.create_odom_message(
                C_k_robot, T_k_inv, timestamp)
            self.odom_pub.publish(odom_msg)
            self.estimates.append(self.C_k.copy())

            self.get_logger().info(
                f"Pose updated | inliers: {len(inliers)} | "
                f"xyz: ({C_k_robot[0,3]:.3f}, "
                f"{C_k_robot[1,3]:.3f}, "
                f"{C_k_robot[2,3]:.3f})")

        except Exception as e:
            self.get_logger().error(f"Error processing frames: {e}")

    # ──────────────────────────────────────────────────────────────
    def match_features(self, img1, img2):
        """ORB detection + FLANN matching with Lowe's ratio test."""
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if (des1 is None or des2 is None
                or len(des1) < 2 or len(des2) < 2):
            self.get_logger().warn("Not enough features detected")
            return None, None, None

        flann   = cv2.FlannBasedMatcher(
            self.index_params, self.search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        self.get_logger().info(f"Good matches: {len(good_matches)}")

        if len(good_matches) < 8:
            return None, None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1, pts2, good_matches

    # ──────────────────────────────────────────────────────────────
    def get_3d_points(self, pts2d, depth_img):
        """Back-project 2D points to 3D using the depth image."""
        pts3d        = []
        valid_indices = []

        # With explicit 32FC1 decoding, scale_factor is always 1.0.
        # uint16 branch kept as fallback if encoding ever changes.
        scale_factor = 1000.0 if depth_img.dtype == np.uint16 else 1.0

        self.get_logger().info(
            f"Depth dtype: {depth_img.dtype}, shape: {depth_img.shape}")

        for i, (u, v) in enumerate(pts2d):
            u_i = int(round(u))
            v_i = int(round(v))
            if (0 <= u_i < depth_img.shape[1] and
                    0 <= v_i < depth_img.shape[0]):
                d = float(depth_img[v_i, u_i]) / scale_factor
                if d <= 0.1 or d > 10.0:
                    continue
                x = (u - self.K[0, 2]) * d / self.K[0, 0]
                y = (v - self.K[1, 2]) * d / self.K[1, 1]
                pts3d.append([x, y, d])
                valid_indices.append(i)

        if len(pts3d) < 4:
            self.get_logger().warn(
                f"Only {len(pts3d)} valid 3D points found")
            return None, None

        return np.array(pts3d, dtype=np.float32), valid_indices

    # ──────────────────────────────────────────────────────────────
    def round_small_values(self, arr, threshold=1e-2):
        """Zero out near-zero values to suppress numerical noise."""
        arr[np.abs(arr) < threshold] = 0.0
        return arr

    # ──────────────────────────────────────────────────────────────
    def create_odom_message(self, T, T_k_inv, timestamp):
        """
        Build a nav_msgs/Odometry from the robot pose matrix T
        (base_link in odom frame) and the incremental transform T_k_inv.
        """
        # Translation — rounding small values is safe here
        translation = self.round_small_values(T[:3, 3].copy())

        # Rotation → quaternion
        # WHY no rounding: small quaternion components are valid
        # (e.g. 1° rotation → component ≈ 0.008). Zeroing them
        # breaks the unit-quaternion constraint and corrupts orientation.
        quaternion = R.from_matrix(T[:3, :3]).as_quat()

        odom                 = Odometry()
        odom.header.stamp    = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        # WHY no axis remapping:
        #   The original code swapped axes (x=z*-1, y=x) as a partial
        #   hack for the missing frame transform. Now that C_k_robot is
        #   correctly expressed in base_link/odom, axes are already right.
        odom.pose.pose.position = Point(
            x=float(translation[0]),
            y=float(translation[1]),
            z=float(translation[2])
        )
        odom.pose.pose.orientation = Quaternion(
            x=float(quaternion[0]),
            y=float(quaternion[1]),
            z=float(quaternion[2]),
            w=float(quaternion[3])
        )

        # ── Velocity ───────────────────────────────────────────────
        # WHY T_k_inv for velocity, not differencing C_k:
        #   T_k_inv[:3,3] is the raw per-frame camera translation,
        #   independent of accumulated drift in C_k.
        #   We rotate it into base_link using only the rotational part
        #   of T_optical_to_base (no translation — velocity has no
        #   offset component, only direction matters).
        if self.last_timestamp is not None:
            dt = ((timestamp.sec  - self.last_timestamp.sec) +
                  (timestamp.nanosec - self.last_timestamp.nanosec)
                  * 1e-9)
            if dt > 0.001:
                R_opt_to_base = self.T_optical_to_base[:3, :3]

                vel_optical = T_k_inv[:3, 3] / dt
                vel_base    = R_opt_to_base @ vel_optical

                odom.twist.twist.linear.x = float(vel_base[0])
                odom.twist.twist.linear.y = float(vel_base[1])
                odom.twist.twist.linear.z = float(vel_base[2])

                rotvec  = R.from_matrix(T_k_inv[:3, :3]).as_rotvec()
                ang_vel = R_opt_to_base @ (rotvec / dt)

                odom.twist.twist.angular.x = float(ang_vel[0])
                odom.twist.twist.angular.y = float(ang_vel[1])
                odom.twist.twist.angular.z = float(ang_vel[2])

        return odom

    # ──────────────────────────────────────────────────────────────
    def visualize_trajectory(self):
        """Save a top-down (X-Z) trajectory plot to ~/ros_ws/."""
        if len(self.estimates) < 2:
            self.get_logger().warn(
                "Not enough poses to visualize trajectory")
            return

        x = [p[0, 3] for p in self.estimates]
        z = [p[2, 3] for p in self.estimates]

        plt.figure(figsize=(10, 5))
        plt.plot(x, z, label="Estimated", color='r')
        plt.legend()
        plt.xlabel("X Position (m)")
        plt.ylabel("Z Position (m)")
        plt.title("Visual OdometrACy Trajectory")
        plt.grid(True)
        path = os.path.expanduser('~/ros_ws/trajectory.png')
        plt.savefig(path)
        self.get_logger().info(f"Trajectory saved to {path}")


# ──────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = OAKVisualOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.visualize_trajectory()
        node.destroy_node()
        # WHY try/except: ROS sometimes shuts down the context before
        # this finally block runs, causing a harmless but noisy RCLError.
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()