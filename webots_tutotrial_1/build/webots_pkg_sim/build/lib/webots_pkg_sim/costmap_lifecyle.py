import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Pose
import numpy as np

class TraversabilityCostmap(LifecycleNode):

    def __init__(self):
        super().__init__('traversability_costmap')
        self.active = False

        # Placeholders
        self.point_cloud_sub = None
        self.costmap_pub = None
        self.inflation_grid_radius = None
        self.grid_resolution = None
        self.grid_size = None
        self.map_center = None
        
        # Parameters
        self.declare_parameter('inflation_grid_radius', 2)
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size', 100)

        self.get_logger().info("Node constructed (unconfigured)")

    def on_configure(self, state):
        self.get_logger().info('Configuring node...')

        # Parameters
        self.inflation_grid_radius = self.get_parameter('inflation_grid_radius').value
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.grid_size = self.get_parameter('grid_size').value

        self.map_center = (self.grid_size * self.grid_resolution) / 2 

        # Publisher
        # here we publish a costmap 2.5D
        self.costmap_pub = self.create_lifecycle_publisher(OccupancyGrid,'/local_costmap',10)

        # Subscription
        # here we subcribe to the pointcloud from the camera so we take from it the points to convert to 3D
        self.point_cloud_sub = self.create_subscription(PointCloud2,'/new_rover/oakd_depth/point_cloud',self.point_cloud_callback,10)

        self.get_logger().info('Configuration complete')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node...')
        self.active = True
        self.costmap_pub.on_activate(state)
        self.get_logger().info('Node is ACTIVE')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating node...')
        self.active = False
        self.costmap_pub.on_deactivate(state)
        self.get_logger().info('Node is INACTIVE')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up resources...')
        if self.costmap_pub:
            self.destroy_publisher(self.costmap_pub)
            self.costmap_pub = None
        if self.point_cloud_sub:
            self.destroy_subscription(self.point_cloud_sub)
            self.point_cloud_sub = None
        self.get_logger().info('Cleanup complete')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info('Shutting down node...')
        self.active = False
        self.get_logger().info('Shutdown complete')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self.get_logger().error('Lifecycle error encountered!')
        self.active = False
        return TransitionCallbackReturn.SUCCESS

    def point_cloud_callback(self, msg):
        #this function publish the costmap 
        if not self.active:
            return
        points = self.extract_points(msg)
        if points is None:
            return
        height_grid = self.build_height_grid(points)
        costmap = self.compute_costmap(height_grid)
        self.publish_costmap(costmap)
    
    def extract_points(self, point_cloud):
        # here we take the points and convert it to x,y,z  >>> 
        points = point_cloud2.read_points(point_cloud, field_names=['x', 'y', 'z'], skip_nans=True)
        if len(points) == 0:
            return None
        X_rob = points['x']
        Y_rob = points['y']
        Z_rob = points['z']
        valid = X_rob > 0
        
        return np.stack((X_rob[valid], Y_rob[valid], Z_rob[valid]), axis=-1)
    
    def build_height_grid(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]


        limit = self.map_center 
        grid_range = [[-limit, limit], [-limit, limit]]
        sum_arr, _, _ = np.histogram2d(x, y, bins=self.grid_size, range=grid_range, weights=z)
        num_arr, _, _ = np.histogram2d(x, y, bins=self.grid_size, range=grid_range)
        height_map = np.divide(sum_arr, num_arr, out=np.full_like(sum_arr, np.nan), where=num_arr!=0).T

        return height_map
    
    def compute_costmap(self, height_map):
        grid_y_indices, grid_x_indices = np.indices((self.grid_size, self.grid_size))
        x_coordinates = (grid_x_indices - self.grid_size/2) * self.grid_resolution
        y_coordinates = (grid_y_indices - self.grid_size/2) * self.grid_resolution

        costmap = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        obstacles_index = []
        
        for grid_y in range(1, self.grid_size-1):
            for grid_x in range(1, self.grid_size-1):

                patch_height = height_map[grid_y-1:grid_y+2, grid_x-1:grid_x+2]
                patch_x = x_coordinates[grid_y-1:grid_y+2, grid_x-1:grid_x+2]
                patch_y = y_coordinates[grid_y-1:grid_y+2, grid_x-1:grid_x+2]

                valid = ~np.isnan(patch_height)
                if np.sum(valid) < 4: continue

                matrix_P = np.column_stack((patch_x[valid], patch_y[valid], patch_height[valid]))
                matrix_A = matrix_P - np.mean(matrix_P, axis=0)

                matrix_Q = matrix_A.T @ matrix_A
                e_values, e_vectors = np.linalg.eigh(matrix_Q)

                normal_vector = e_vectors[:, 0]
                dot_product = abs(normal_vector @ np.array([0,0,1]).T)
                slope = np.arccos(np.clip(dot_product, -1.0, 1.0))
                slope_deg = slope*180/np.pi
                roughness = np.sqrt(abs(e_values[0])) 
                height_diff = np.max(patch_height[valid]) - np.min(patch_height[valid])

                if (slope_deg >= 25 and height_diff >= 0.08) or height_diff >= 0.15 or roughness >= 0.05:
                    costmap[grid_y, grid_x] = 100
                    obstacles_index.append((grid_y, grid_x))
                elif height_diff >= 0.10:
                    costmap[grid_y, grid_x] = 80
                elif height_diff >= 0.05:
                    costmap[grid_y, grid_x] = 60
                elif height_diff >= 0.03:
                    costmap[grid_y, grid_x] = 30                     
                else:
                    costmap[grid_y, grid_x] = 0

        costmap = self.apply_inflation(costmap, obstacles_index)

        return costmap

    def apply_inflation(self, costmap, obstacles_index):
        r = self.inflation_grid_radius
        for obs_y, obs_x in obstacles_index:
            for dy in range(-r, r+1):
                for dx in range(-r,r+1):
                    nx, ny = obs_x + dx, obs_y + dy
                    dist = np.sqrt(dx**2+dy**2)
                    if dist <=r:
                        cost = int(max(0, 100-(20*dist)))
                        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                            costmap[ny, nx] = max(cost, costmap[ny, nx])
        return costmap


    def publish_costmap(self, grid):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.info.resolution = self.grid_resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size

        msg.info.origin = Pose()
        msg.info.origin.position.x = -self.map_center
        msg.info.origin.position.y = -self.map_center
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten().astype(np.int8).tolist()
        self.costmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityCostmap()
    rclpy.spin(node)
    rclpy.shutdown()

if '__name__' == main:
    main()