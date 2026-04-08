import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

HALF_DISTANCE_BETWEEN_WHEELS = 0.06
WHEEL_RADIUS = 0.03

class RobotDriver:
    def init(self, webots_node, properties):
        self._target_twist = Twist()
        self._motors = {}
        self._motor_names = ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']

        self._robot = webots_node.robot

        rclpy.init(args=None)  # FIX: must initialize rclpy before creating a node
        self._node = rclpy.create_node('robot_driver')

        for name in self._motor_names:
            motor = self._robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
                self._motors[name] = motor
            else:
                self._node.get_logger().warn(f'Motor not found: {name}')

        self._node.create_subscription(Twist, 'cmd_vel', self._cmd_vel_callback, 1)
        self._joint_state_publisher = self._node.create_publisher(JointState, 'joint_states', 1)
        self._node.get_logger().info('RobotDriver initialized successfully')

    def _cmd_vel_callback(self, twist):
        self._target_twist = twist

    def step(self):
        if not self._motors:
            return

        rclpy.spin_once(self._node, timeout_sec=0)

        forward_speed = self._target_twist.linear.x
        angular_speed = self._target_twist.angular.z

        v_left  = (forward_speed - angular_speed * HALF_DISTANCE_BETWEEN_WHEELS) / WHEEL_RADIUS
        v_right = (forward_speed + angular_speed * HALF_DISTANCE_BETWEEN_WHEELS) / WHEEL_RADIUS

        self._motors['fl_wheel_joint'].setVelocity(v_left)
        self._motors['rl_wheel_joint'].setVelocity(v_left)
        self._motors['fr_wheel_joint'].setVelocity(v_right)
        self._motors['rr_wheel_joint'].setVelocity(v_right)

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = self._motor_names
        msg.velocity = [v_left, v_right, v_left, v_right]
        msg.position = [0.0, 0.0, 0.0, 0.0]
        self._joint_state_publisher.publish(msg)