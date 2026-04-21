import rclpy
from geometry_msgs.msg import TwistStamped, TransformStamped
from sensor_msgs.msg import JointState, Image, Imu, CameraInfo, NavSatFix
from tf2_ros import StaticTransformBroadcaster
import math
import struct

# ── Robot geometry ─────────────────────────────────────────────
WHEEL_RADIUS    = 0.086
WHEEL_BASE      = 0.5898   # front-to-rear distance (Y axis)
TRACK_WIDTH     = 0.6934   # left-to-right distance  (X axis)
MAX_STEER_ANGLE = 1.5708   # ✅ SWERVE: full 360° steering (π/2 = 90°)
                            #    change to math.pi for full rotation

WHEEL_JOINTS    = ["wheel1_joint", "wheel2_joint",
                   "wheel3_joint", "wheel4_joint"]
STEERING_JOINTS = ["steering1_joint", "steering2_joint",
                   "steering3_joint", "steering4_joint"]
ALL_JOINTS      = WHEEL_JOINTS + STEERING_JOINTS

# Wheel positions relative to robot center [x, y]
# steering1/wheel1: front-right (+x, +y)
# steering2/wheel2: front-left  (-x, +y)
# steering3/wheel3: rear-right  (+x, -y)
# steering4/wheel4: rear-left   (-x, -y)
WHEEL_POSITIONS = {
    "wheel1_joint": ( TRACK_WIDTH/2,  WHEEL_BASE/2),
    "wheel2_joint": (-TRACK_WIDTH/2,  WHEEL_BASE/2),
    "wheel3_joint": ( TRACK_WIDTH/2, -WHEEL_BASE/2),
    "wheel4_joint": (-TRACK_WIDTH/2, -WHEEL_BASE/2),
}

WHEEL_AXIS_SIGN = {
    "wheel1_joint":  1,
    "wheel2_joint":  1,
    "wheel3_joint": -1,
    "wheel4_joint": -1,
}

class RoverDriverSwerve:
    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        self._lin_x = 0.0   # forward/backward
        self._lin_y = 0.0   # strafe left/right  ✅ SWERVE: adds lateral movement
        self._ang   = 0.0   # rotation

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

        self._imu   = self._robot.getDevice("imu")
        self._accel = self._robot.getDevice("accelerometer")
        self._gyro  = self._robot.getDevice("gyro")
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
        self._pub_joints     = self._node.create_publisher(
            JointState, "joint_states", 1)
        self._pub_imu        = self._node.create_publisher(
            Imu, "imu", 10)
        self._pub_gps        = self._node.create_publisher(
            NavSatFix, "gps/fix", 10)
        self._pub_rgb        = self._node.create_publisher(
            Image, "camera/image_raw", 10)
        self._pub_depth      = self._node.create_publisher(
            Image, "camera/depth/image_raw", 10)
        self._pub_cam_info   = self._node.create_publisher(
            CameraInfo, "camera/camera_info", 10)
        self._pub_depth_info = self._node.create_publisher(
            CameraInfo, "camera/depth/camera_info", 10)

        # ── ✅ FIX: Static TF broadcaster ──────────────────────
        # Publishes fixed transforms between camera frames and base_link
        # This fixes: "Could not transform from [oakd_depth] to [base_link]"
        self._tf_broadcaster = StaticTransformBroadcaster(self._node)
        self._publish_static_transforms()

        self._node.get_logger().info("RoverDriver (Swerve) ready")

    # ──────────────────────────────────────────────────────────
    def _publish_static_transforms(self):
        """
        Publish static TF transforms between all camera frames and base_link.
        These are fixed transforms that never change.
        """
        transforms = []
        stamp = self._node.get_clock().now().to_msg()

        # ── base_link → oakd_link (camera body) ───────────────
        t = TransformStamped()
        t.header.stamp         = stamp
        t.header.frame_id      = "base_link"
        t.child_frame_id       = "oakd_link"
        t.transform.translation.x = 0.4    # 40cm in front
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.15   # 15cm up
        t.transform.rotation.w    = 1.0    # no rotation
        transforms.append(t)

        # ── oakd_link → oakd_rgb_optical_link ─────────────────
        t2 = TransformStamped()
        t2.header.stamp         = stamp
        t2.header.frame_id      = "oakd_link"
        t2.child_frame_id       = "oakd_rgb_optical_link"
        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        # Optical frame: rotate -90° around Z then -90° around X
        # This converts from ROS camera convention to optical convention
        t2.transform.rotation.x = -0.5
        t2.transform.rotation.y =  0.5
        t2.transform.rotation.z = -0.5
        t2.transform.rotation.w =  0.5
        transforms.append(t2)

        # ── oakd_link → oakd_depth_optical_link ───────────────
        # Same position as RGB (aligned depth camera)
        t3 = TransformStamped()
        t3.header.stamp         = stamp
        t3.header.frame_id      = "oakd_link"
        t3.child_frame_id       = "oakd_depth_optical_link"
        t3.transform.translation.x = 0.0
        t3.transform.translation.y = 0.0
        t3.transform.translation.z = 0.0
        t3.transform.rotation.x = -0.5
        t3.transform.rotation.y =  0.5
        t3.transform.rotation.z = -0.5
        t3.transform.rotation.w =  0.5
        transforms.append(t3)

        # ── oakd_link → oakd_depth (Webots frame name) ────────
        # ✅ FIX: This is the missing transform causing the error
        t4 = TransformStamped()
        t4.header.stamp         = stamp
        t4.header.frame_id      = "oakd_link"
        t4.child_frame_id       = "oakd_depth"
        t4.transform.translation.x = 0.0
        t4.transform.translation.y = 0.0
        t4.transform.translation.z = 0.0
        t4.transform.rotation.x = -0.5
        t4.transform.rotation.y =  0.5
        t4.transform.rotation.z = -0.5
        t4.transform.rotation.w =  0.5
        transforms.append(t4)

        # ── base_link → imu_link ───────────────────────────────
        t5 = TransformStamped()
        t5.header.stamp         = stamp
        t5.header.frame_id      = "base_link"
        t5.child_frame_id       = "imu_link"
        t5.transform.translation.x = 0.0
        t5.transform.translation.y = 0.0
        t5.transform.translation.z = 0.1
        t5.transform.rotation.w    = 1.0
        transforms.append(t5)

        # ── base_link → gps_link ───────────────────────────────
        t6 = TransformStamped()
        t6.header.stamp         = stamp
        t6.header.frame_id      = "base_link"
        t6.child_frame_id       = "gps_link"
        t6.transform.translation.x = 0.0
        t6.transform.translation.y = 0.0
        t6.transform.translation.z = 0.22
        t6.transform.rotation.w    = 1.0
        transforms.append(t6)

        self._tf_broadcaster.sendTransform(transforms)
        self._node.get_logger().info(
            "Static TF transforms published: "
            "base_link → oakd_link → oakd_depth/rgb frames")

    # ──────────────────────────────────────────────────────────
    def _cmd_vel_cb(self, msg):
        self._lin_x = max(-0.5, min(0.5, msg.twist.linear.x))
        self._lin_y = max(-0.5, min(0.5, msg.twist.linear.y))   # ✅ SWERVE: strafe
        self._ang   = max(-0.3, min(0.3, msg.twist.angular.z))

    # ──────────────────────────────────────────────────────────
    def _compute_swerve(self):
        """
        ✅ SWERVE DRIVE KINEMATICS

        For each wheel, calculate:
        1. The desired velocity vector (speed + direction)
        2. Convert to steering angle + wheel speed

        Swerve formula for each wheel:
          vx_wheel = lin_x - ang * wheel_y
          vy_wheel = lin_y + ang * wheel_x
          speed    = sqrt(vx^2 + vy^2)
          angle    = atan2(vy, vx)

        This allows:
          - Forward/backward movement
          - Strafing (sideways movement) ← unique to swerve
          - Rotation in place
          - Diagonal movement
          - All combinations simultaneously
        """
        wheel_cmds = {}

        for wheel_name, steer_name in zip(WHEEL_JOINTS, STEERING_JOINTS):
            wx, wy = WHEEL_POSITIONS[wheel_name]

            # Velocity components for this wheel
            vx = self._lin_x - self._ang * wy
            vy = self._lin_y + self._ang * wx

            # Raw speed and angle
            speed = math.sqrt(vx**2 + vy**2)
            angle = math.atan2(vy, vx) if speed > 0.01 else 0.0

            # ── Key fix: fold angle into [-π/2, π/2] by reversing wheel ──
            # This avoids needing >90° steering travel.
            if angle > math.pi / 2:
                angle -= math.pi
                speed = -speed          # run wheel backwards
            elif angle < -math.pi / 2:
                angle += math.pi
                speed = -speed          # run wheel backwards

            wheel_cmds[wheel_name] = speed / WHEEL_RADIUS
            wheel_cmds[steer_name] = angle

        return wheel_cmds

    # ──────────────────────────────────────────────────────────
    def _make_camera_info(self, stamp, frame_id, w, h):
        """Build CameraInfo from camera dimensions."""
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

    

    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)
        stamp = self._node.get_clock().now().to_msg()

        # ── Swerve: compute wheel commands ────────────────────────
        cmds = self._compute_swerve()

        # Apply steering positions
        for name in STEERING_JOINTS:
            self._steers[name].setPosition(cmds[name])

        # Apply wheel velocities with per-wheel axis correction.
        # cmds[name] is already signed (negative = reverse direction),
        # so we must NOT unconditionally negate wheels 3/4 as before.
        print(cmds)
        for name in WHEEL_JOINTS:
            self._wheels[name].setVelocity(
                cmds[name] * WHEEL_AXIS_SIGN[name]
            )

        # ── Publish IMU ────────────────────────────────────────────
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
                0.0,  0.01, 0.0,
                0.0,  0.0,  0.01]
            imu_msg.angular_velocity_covariance = [
                4e-8, 0.0,  0.0,
                0.0,  4e-8, 0.0,
                0.0,  0.0,  4e-8]
            imu_msg.linear_acceleration_covariance = [
                3e-4, 0.0,  0.0,
                0.0,  3e-4, 0.0,
                0.0,  0.0,  3e-4]
            self._pub_imu.publish(imu_msg)
        except Exception as e:
            self._node.get_logger().warn(
                f"IMU error: {e}", throttle_duration_sec=5)

        # ── Publish GPS ────────────────────────────────────────────
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
                0.25, 0.0,  0.0,
                0.0,  0.25, 0.0,
                0.0,  0.0,  0.25]
            gps_msg.position_covariance_type = 2
            self._pub_gps.publish(gps_msg)
        except Exception as e:
            self._node.get_logger().warn(
                f"GPS error: {e}", throttle_duration_sec=5)

        # ── Publish RGB camera ─────────────────────────────────────
        if self._camera:
            raw = self._camera.getImage()
            if raw:
                w = self._camera.getWidth()
                h = self._camera.getHeight()
                img_msg = Image()
                img_msg.header.stamp    = stamp
                img_msg.header.frame_id = "oakd_rgb_optical_link"
                img_msg.width    = w
                img_msg.height   = h
                img_msg.encoding = "bgra8"
                img_msg.step     = w * 4
                img_msg.data     = list(raw)
                self._pub_rgb.publish(img_msg)

                cam_info = self._make_camera_info(
                    stamp, "oakd_rgb_optical_link", w, h)
                self._pub_cam_info.publish(cam_info)

        # ── Publish depth camera ───────────────────────────────────
        if self._depth:
            raw_d = self._depth.getRangeImage()
            if raw_d:
                w = self._depth.getWidth()
                h = self._depth.getHeight()
                data = struct.pack(f"{w*h}f", *raw_d)

                depth_msg = Image()
                depth_msg.header.stamp    = stamp
                depth_msg.header.frame_id = "oakd_depth_optical_link"
                depth_msg.width    = w
                depth_msg.height   = h
                depth_msg.encoding = "32FC1"
                depth_msg.step     = w * 4
                depth_msg.data     = list(data)
                self._pub_depth.publish(depth_msg)

                depth_info = self._make_camera_info(
                    stamp, "oakd_depth_optical_link", w, h)
                self._pub_depth_info.publish(depth_info)

        # ── Publish joint states ───────────────────────────────────
        js = JointState()
        js.header.stamp = stamp
        js.name         = ALL_JOINTS
        js.velocity     = [0.0] * 8
        js.position     = [0.0] * 8
        self._pub_joints.publish(js)