import rclpy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Image, Imu, CameraInfo, NavSatFix
import math
import struct

WHEEL_RADIUS    = 0.086
WHEEL_BASE      = 0.5898
TRACK_WIDTH     = 0.6934
MAX_STEER_ANGLE = 0.5

WHEEL_JOINTS    = ["wheel1_joint", "wheel2_joint",
                   "wheel3_joint", "wheel4_joint"]
STEERING_JOINTS = ["steering1_joint", "steering2_joint",
                   "steering3_joint", "steering4_joint"]
ALL_JOINTS      = WHEEL_JOINTS + STEERING_JOINTS


class RoverDriver:
    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        self._lin   = 0.0
        self._ang   = 0.0

        rclpy.init(args=None)
        self._node = rclpy.create_node("rover_driver")

        # ── Wheel motors ───────────────────────────────────────
        self._wheels = {}
        for name in WHEEL_JOINTS:
            m = self._robot.getDevice(name)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self._wheels[name] = m

        # ── Steering motors ────────────────────────────────────
        self._steers = {}
        for name in STEERING_JOINTS:
            m = self._robot.getDevice(name)
            m.setPosition(0.0)
            self._steers[name] = m

        # ── Sensors ────────────────────────────────────────────
        timestep = int(self._robot.getBasicTimeStep())

        self._imu = self._robot.getDevice("imu")
        self._accel = self._robot.getDevice("accelerometer")
        self._gyro = self._robot.getDevice("gyro")
        self._imu.enable(timestep)
        self._accel.enable(timestep)
        self._gyro.enable(timestep)

        self._gps = self._robot.getDevice("gps")
        self._gps.enable(timestep)

        self._camera = self._robot.getDevice("oakd_rgb")
        if self._camera:
            self._camera.enable(timestep)

        self._depth = self._robot.getDevice("oakd_depth")
        if self._depth:
            self._depth.enable(timestep)

        # ── ROS subscribers ────────────────────────────────────
        self._node.create_subscription(
            TwistStamped, "cmd_vel", self._cmd_vel_cb, 1)

        # ── ROS publishers ─────────────────────────────────────
        self._pub_joints = self._node.create_publisher(
            JointState, "joint_states", 1)
        self._pub_imu = self._node.create_publisher(
            Imu, "imu", 10)
        self._pub_gps = self._node.create_publisher(
            NavSatFix, "gps/fix", 10)
        self._pub_rgb = self._node.create_publisher(
            Image, "camera/image_raw", 10)
        self._pub_depth = self._node.create_publisher(
            Image, "camera/depth/image_raw", 10)
        self._pub_cam_info = self._node.create_publisher(
            CameraInfo, "camera/camera_info", 10)
        self._pub_depth_info = self._node.create_publisher(
            CameraInfo, "camera/depth/camera_info", 10)

        self._node.get_logger().info("RoverDriver ready")

    # ── Callbacks ──────────────────────────────────────────────

    def _cmd_vel_cb(self, msg):
        self._lin = max(-0.5, min(0.5, msg.twist.linear.x))
        self._ang = max(-0.3, min(0.3, msg.twist.angular.z))

    # ── Helpers ────────────────────────────────────────────────

    def _steer(self, s1, s2, s3, s4):
        self._steers["steering1_joint"].setPosition(
            max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, s1)))
        self._steers["steering2_joint"].setPosition(
            max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, s2)))
        self._steers["steering3_joint"].setPosition(
            max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, s3)))
        self._steers["steering4_joint"].setPosition(
            max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, s4)))

    def _drive(self, w1, w2, w3, w4):
        # wheel3/4 axis is 0 -1 0 so negate
        self._wheels["wheel1_joint"].setVelocity( w1)
        self._wheels["wheel2_joint"].setVelocity( w2)
        self._wheels["wheel3_joint"].setVelocity(-w3)
        self._wheels["wheel4_joint"].setVelocity(-w4)

    def _make_camera_info(self, stamp, frame_id, w, h):
        """Build a CameraInfo message from camera dimensions."""
        # Approximate focal length from FOV=1.0472 rad (60 degrees)
        fx = w / (2.0 * math.tan(1.0472 / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        msg = CameraInfo()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.width  = w
        msg.height = h
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx,  0.0, cx,
                 0.0, fy,  cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [fx,  0.0, cx,  0.0,
                 0.0, fy,  cy,  0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    # ── Main step ──────────────────────────────────────────────

    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)
        stamp = self._node.get_clock().now().to_msg()
        lin = self._lin
        ang = self._ang

        # ── Motion control ─────────────────────────────────────
        if abs(lin) > 0.01:
            # DRIVING: front wheels steer, rear straight
            if abs(ang) > 0.01:
                steer = math.atan(ang * WHEEL_BASE / (2.0 * abs(lin)))
                steer = max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, steer))
            else:
                steer = 0.0
            self._steer(steer, -steer, 0.0, 0.0)
            v_pos_x = (lin - ang * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            v_neg_x = (lin + ang * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            self._drive(v_pos_x, v_neg_x, v_pos_x, v_neg_x)

        elif abs(ang) > 0.01:
            # ROTATE: differential drive
            self._steer(0.0, 0.0, 0.0, 0.0)
            v = abs(ang) * TRACK_WIDTH / (2.0 * WHEEL_RADIUS)
            v = max(v, 1.0)
            if ang > 0:
                self._drive(-v,  v, -v,  v)
            else:
                self._drive( v, -v,  v, -v)

        else:
            # STOP
            self._steer(0.0, 0.0, 0.0, 0.0)
            self._drive(0.0, 0.0, 0.0, 0.0)

        # ── Publish IMU ────────────────────────────────────────
        try:
            rpy   = self._imu.getRollPitchYaw()
            gyro  = self._gyro.getValues()
            accel = self._accel.getValues()

            r_, p_, y_ = rpy
            cr, sr = math.cos(r_/2), math.sin(r_/2)
            cp, sp = math.cos(p_/2), math.sin(p_/2)
            cy, sy = math.cos(y_/2), math.sin(y_/2)

            imu_msg = Imu()
            imu_msg.header.stamp    = stamp
            imu_msg.header.frame_id = "imu_link"
            imu_msg.orientation.w   = cr*cp*cy + sr*sp*sy
            imu_msg.orientation.x   = sr*cp*cy - cr*sp*sy
            imu_msg.orientation.y   = cr*sp*cy + sr*cp*sy
            imu_msg.orientation.z   = cr*cp*sy - sr*sp*cy
            imu_msg.angular_velocity.x    = float(gyro[0])
            imu_msg.angular_velocity.y    = float(gyro[1])
            imu_msg.angular_velocity.z    = float(gyro[2])
            imu_msg.linear_acceleration.x = float(accel[0])
            imu_msg.linear_acceleration.y = float(accel[1])
            imu_msg.linear_acceleration.z = float(accel[2])
            imu_msg.orientation_covariance = [
                0.01, 0.0, 0.0,
                0.0, 0.01, 0.0,
                0.0, 0.0, 0.01]
            imu_msg.angular_velocity_covariance = [
                4e-8, 0.0, 0.0,
                0.0, 4e-8, 0.0,
                0.0, 0.0, 4e-8]
            imu_msg.linear_acceleration_covariance = [
                3e-4, 0.0, 0.0,
                0.0, 3e-4, 0.0,
                0.0, 0.0, 3e-4]
            self._pub_imu.publish(imu_msg)
        except Exception as e:
            self._node.get_logger().warn(f"IMU error: {e}", throttle_duration_sec=5)

        # ── Publish GPS ────────────────────────────────────────
        try:
            vals = self._gps.getValues()
            gps_msg = NavSatFix()
            gps_msg.header.stamp    = stamp
            gps_msg.header.frame_id = "gps_link"
            gps_msg.latitude  = float(vals[0])
            gps_msg.longitude = float(vals[1])
            gps_msg.altitude  = float(vals[2])
            gps_msg.status.status  = 0
            gps_msg.status.service = 1
            gps_msg.position_covariance = [
                0.25, 0.0, 0.0,
                0.0, 0.25, 0.0,
                0.0, 0.0, 0.25]
            gps_msg.position_covariance_type = 2
            self._pub_gps.publish(gps_msg)
        except Exception as e:
            self._node.get_logger().warn(f"GPS error: {e}", throttle_duration_sec=5)

        # ── Publish RGB camera ─────────────────────────────────
        if self._camera:
            raw = self._camera.getImage()
            if raw:
                w = self._camera.getWidth()
                h = self._camera.getHeight()

                # Publish image
                img_msg = Image()
                img_msg.header.stamp    = stamp
                img_msg.header.frame_id = "oakd_rgb_optical_link"
                img_msg.width    = w
                img_msg.height   = h
                img_msg.encoding = "bgra8"
                img_msg.step     = w * 4
                img_msg.data     = list(raw)
                self._pub_rgb.publish(img_msg)

                # Publish camera_info (needed by VO node)
                cam_info = self._make_camera_info(
                    stamp, "oakd_rgb_optical_link", w, h)
                self._pub_cam_info.publish(cam_info)

        # ── Publish depth camera ───────────────────────────────
        if self._depth:
            raw_d = self._depth.getRangeImage()
            if raw_d:
                w = self._depth.getWidth()
                h = self._depth.getHeight()
                data = struct.pack(f"{w*h}f", *raw_d)

                # Publish depth image
                depth_msg = Image()
                depth_msg.header.stamp    = stamp
                depth_msg.header.frame_id = "oakd_depth_optical_link"
                depth_msg.width    = w
                depth_msg.height   = h
                depth_msg.encoding = "32FC1"
                depth_msg.step     = w * 4
                depth_msg.data     = list(data)
                self._pub_depth.publish(depth_msg)

                # Publish depth camera_info (needed by VO node)
                depth_info = self._make_camera_info(
                    stamp, "oakd_depth_optical_link", w, h)
                self._pub_depth_info.publish(depth_info)

        # ── Publish joint states ───────────────────────────────
        js = JointState()
        js.header.stamp = stamp
        js.name         = ALL_JOINTS
        js.velocity     = [0.0] * 8
        js.position     = [0.0] * 8
        self._pub_joints.publish(js)