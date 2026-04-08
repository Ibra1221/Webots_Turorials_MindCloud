import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import math
import sys
import tty
import termios
import threading

WHEEL_RADIUS    = 0.086
WHEEL_BASE      = 0.5898
TRACK_WIDTH     = 0.6934
MAX_STEER_ANGLE = 1.2
LINEAR_SPEED    = 0.5
ANGULAR_SPEED   = 1.0

WHEEL_JOINTS    = ['wheel1_joint', 'wheel2_joint', 'wheel3_joint', 'wheel4_joint']
STEERING_JOINTS = ['steering1_joint', 'steering2_joint', 'steering3_joint', 'steering4_joint']
ALL_JOINTS      = WHEEL_JOINTS + STEERING_JOINTS

KEY_BINDINGS = {
    'i': ( 1,  0),   # forward
    ',': (-1,  0),   # backward
    'j': ( 0,  1),   # turn left
    'l': ( 0, -1),   # turn right
    'u': ( 1,  1),   # forward + left
    'o': ( 1, -1),   # forward + right
    'm': (-1, -1),   # backward + left
    '.': (-1,  1),   # backward + right
    'k': ( 0,  0),   # stop
}

class RoverDriver:
    def init(self, webots_node, properties):
        self._robot        = webots_node.robot
        self._lin          = 0.0
        self._ang          = 0.0

        rclpy.init(args=None)
        self._node = rclpy.create_node('rover_driver')

        # Wheel motors — velocity control
        self._wheels = {}
        for name in WHEEL_JOINTS:
            m = self._robot.getDevice(name)
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            self._wheels[name] = m

        # Steering motors — position control
        self._steers = {}
        for name in STEERING_JOINTS:
            m = self._robot.getDevice(name)
            m.setPosition(0.0)
            self._steers[name] = m

        # ROS subscription (external teleop still works)
        self._node.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 1)
        self._pub = self._node.create_publisher(JointState, 'joint_states', 1)

        # Start keyboard thread
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()

        self._node.get_logger().info('RoverDriver ready — use i/j/k/l/, to drive')

    def _cmd_vel_cb(self, msg):
        # External cmd_vel overrides keyboard
        self._lin = msg.linear.x
        self._ang = msg.angular.z

    def _get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _keyboard_loop(self):
        self._node.get_logger().info('Keyboard ready: i=fwd  ,=back  j=left  l=right  k=stop')
        while True:
            try:
                key = self._get_key()
                if key in KEY_BINDINGS:
                    lin_dir, ang_dir = KEY_BINDINGS[key]
                    self._lin = lin_dir * LINEAR_SPEED
                    self._ang = ang_dir * ANGULAR_SPEED
                elif key == '\x03':  # Ctrl+C
                    break
            except Exception:
                break

    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)

        lin = self._lin
        ang = self._ang

        # Steering
        if abs(ang) > 0.01:
            if abs(lin) > 0.001:
                steer = math.atan(ang * WHEEL_BASE / (2.0 * abs(lin)))
            else:
                steer = math.copysign(MAX_STEER_ANGLE, ang)
        else:
            steer = 0.0

        steer = max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, steer))

        self._steers['steering1_joint'].setPosition( steer)
        self._steers['steering2_joint'].setPosition( steer)
        self._steers['steering3_joint'].setPosition(-steer)
        self._steers['steering4_joint'].setPosition(-steer)

        # Wheel speeds
        left  = (lin - ang * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        right = (lin + ang * TRACK_WIDTH / 2.0) / WHEEL_RADIUS

        self._wheels['wheel1_joint'].setVelocity(left)
        self._wheels['wheel3_joint'].setVelocity(left)
        self._wheels['wheel2_joint'].setVelocity(right)
        self._wheels['wheel4_joint'].setVelocity(right)

        # Joint states
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name         = ALL_JOINTS
        msg.velocity     = [left, right, left, right, 0.0, 0.0, 0.0, 0.0]
        msg.position     = [0.0] * 8
        self._pub.publish(msg)
