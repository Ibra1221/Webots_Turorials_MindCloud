#!/usr/bin/env python3
"""
2.5D Traversability Costmap Node

How it works:
1. Receives PointCloud2 from depth camera
2. Builds a 2D height grid (each cell = average height)
3. For each 3x3 patch: calculates slope, roughness, height difference
4. Assigns traversability cost (0=free, 100=obstacle)
5. Applies inflation around obstacles
6. Publishes as OccupancyGrid → /local_costmap

Cost values:
  -1  = unknown (no depth data)
   0  = free (flat, safe)
  30  = slight roughness
  60  = moderate roughness
  80  = steep
  100 = obstacle (too steep or too rough)
"""

import rclpy
from rclpy.node import Node          # ✅ FIX: use regular Node not LifecycleNode

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Pose
import numpy as np


class TraversabilityCostmap(Node):

    def __init__(self):
        super().__init__('traversability_costmap')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('inflation_grid_radius', 2)
        self.declare_parameter('grid_resolution', 0.1)   # metres per cell
        self.declare_parameter('grid_size', 100)          # 100x100 grid = 10x10 metres
        # ✅ FIX: make pointcloud topic configurable
        self.declare_parameter('pointcloud_topic', '/new_rover/oakd_depth/point_cloud')
        self.declare_parameter('frame_id', 'base_link')

        self.inflation_grid_radius = self.get_parameter('inflation_grid_radius').value
        self.grid_resolution       = self.get_parameter('grid_resolution').value
        self.grid_size             = self.get_parameter('grid_size').value
        pointcloud_topic           = self.get_parameter('pointcloud_topic').value
        self.frame_id              = self.get_parameter('frame_id').value

        # Half the grid size in metres (used for centering the map)
        self.map_center = (self.grid_size * self.grid_resolution) / 2

        # ── Publisher ──────────────────────────────────────────────
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap', 10)

        # ── Subscriber ─────────────────────────────────────────────
        # ✅ FIX: use correct topic
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.point_cloud_callback,
            10)

        self.get_logger().info(
            f'Traversability costmap ready\n'
            f'  pointcloud: {pointcloud_topic}\n'
            f'  grid: {self.grid_size}x{self.grid_size} '
            f'@ {self.grid_resolution}m/cell\n'
            f'  area: {self.grid_size * self.grid_resolution}x'
            f'{self.grid_size * self.grid_resolution}m')

    # ──────────────────────────────────────────────────────────────
    def point_cloud_callback(self, msg):
        """Main pipeline: pointcloud → height grid → costmap → publish."""
        points = self.extract_points(msg)
        if points is None:
            return

        height_grid = self.build_height_grid(points)
        costmap     = self.compute_costmap(height_grid)
        self.publish_costmap(costmap)

    # ──────────────────────────────────────────────────────────────
    def extract_points(self, point_cloud):
        """
        Extract valid XYZ points from PointCloud2.
        Filters out NaN values and points behind the camera.
        """
        points = point_cloud2.read_points(
            point_cloud,
            field_names=['x', 'y', 'z'],
            skip_nans=True)

        if len(points) == 0:
            self.get_logger().warn(
                'Empty pointcloud', throttle_duration_sec=2)
            return None

        X = points['x']
        Y = points['y']
        Z = points['z']

        # Keep only points in front of camera (x > 0)
        # and within reasonable range
        valid = (X > 0.1) & (X < 10.0) & np.isfinite(Z)

        if np.sum(valid) < 10:
            self.get_logger().warn(
                f'Too few valid points: {np.sum(valid)}',
                throttle_duration_sec=2)
            return None

        return np.stack((X[valid], Y[valid], Z[valid]), axis=-1)

    # ──────────────────────────────────────────────────────────────
    def build_height_grid(self, points):
        """
        Build a 2D grid where each cell contains the average height.

        Think of it as looking at the ground from above:
        each grid cell = average Z of all points in that area.

        NaN = no points in that cell (unknown)
        """
        x = points[:, 0]  # forward
        y = points[:, 1]  # sideways
        z = points[:, 2]  # height

        limit      = self.map_center
        grid_range = [[-limit, limit], [-limit, limit]]

        # Sum of heights in each cell
        sum_arr, _, _ = np.histogram2d(
            x, y,
            bins=self.grid_size,
            range=grid_range,
            weights=z)

        # Count of points in each cell
        num_arr, _, _ = np.histogram2d(
            x, y,
            bins=self.grid_size,
            range=grid_range)

        # Average height = sum / count (NaN where no points)
        height_map = np.divide(
            sum_arr, num_arr,
            out=np.full_like(sum_arr, np.nan),
            where=num_arr != 0).T

        return height_map

    # ──────────────────────────────────────────────────────────────
    def compute_costmap(self, height_map):
        """
        Analyze each 3x3 patch of the height grid.

        For each patch we calculate:
        1. slope_deg  — how tilted is the surface? (degrees)
        2. roughness  — how irregular is the surface?
        3. height_diff — max height variation in patch

        Then assign cost based on thresholds.
        """
        costmap        = np.full(
            (self.grid_size, self.grid_size), -1, dtype=np.int8)
        obstacles_index = []

        # Pre-compute coordinate grids
        grid_y_idx, grid_x_idx = np.indices(
            (self.grid_size, self.grid_size))
        x_coords = (grid_x_idx - self.grid_size / 2) * self.grid_resolution
        y_coords = (grid_y_idx - self.grid_size / 2) * self.grid_resolution

        for gy in range(1, self.grid_size - 1):
            for gx in range(1, self.grid_size - 1):

                # Get 3x3 patch of heights and coordinates
                patch_h = height_map[gy-1:gy+2, gx-1:gx+2]
                patch_x = x_coords[gy-1:gy+2, gx-1:gx+2]
                patch_y = y_coords[gy-1:gy+2, gx-1:gx+2]

                # Need at least 4 valid points to fit a plane
                valid = ~np.isnan(patch_h)
                if np.sum(valid) < 4:
                    continue  # stays -1 (unknown)

                # ── PCA to find surface normal ─────────────────
                # Stack valid points into Nx3 matrix
                P = np.column_stack((
                    patch_x[valid],
                    patch_y[valid],
                    patch_h[valid]))

                # Center the points
                A = P - np.mean(P, axis=0)

                # Covariance-like matrix
                Q = A.T @ A

                # Smallest eigenvalue → normal vector direction
                e_values, e_vectors = np.linalg.eigh(Q)
                normal = e_vectors[:, 0]  # smallest eigenvalue

                # ── Calculate slope ────────────────────────────
                # Angle between surface normal and vertical (0,0,1)
                dot      = abs(normal @ np.array([0, 0, 1]))
                slope    = np.arccos(np.clip(dot, -1.0, 1.0))
                slope_deg = slope * 180 / np.pi

                # ── Calculate roughness ────────────────────────
                # Sqrt of smallest eigenvalue = surface irregularity
                roughness = np.sqrt(abs(e_values[0]))

                # ── Calculate height difference ────────────────
                h_diff = (np.max(patch_h[valid]) -
                          np.min(patch_h[valid]))

                # ── Assign cost ────────────────────────────────
                if ((slope_deg >= 25 and h_diff >= 0.08) or
                        h_diff >= 0.15 or roughness >= 0.05):
                    costmap[gy, gx] = 100   # obstacle
                    obstacles_index.append((gy, gx))
                elif h_diff >= 0.10:
                    costmap[gy, gx] = 80    # steep
                elif h_diff >= 0.05:
                    costmap[gy, gx] = 60    # rough
                elif h_diff >= 0.03:
                    costmap[gy, gx] = 30    # slight
                else:
                    costmap[gy, gx] = 0     # free

        # Apply inflation around obstacles
        costmap = self.apply_inflation(costmap, obstacles_index)

        # Log statistics
        n_free     = np.sum(costmap == 0)
        n_obstacle = np.sum(costmap == 100)
        n_unknown  = np.sum(costmap == -1)
        self.get_logger().info(
            f'Costmap: free={n_free} obstacle={n_obstacle} '
            f'unknown={n_unknown}',
            throttle_duration_sec=2)

        return costmap

    # ──────────────────────────────────────────────────────────────
    def apply_inflation(self, costmap, obstacles_index):
        """
        Inflate obstacles — makes robot stay away from edges.

        For each obstacle cell, spread cost to nearby cells:
        distance 0 → cost 100
        distance 1 → cost 80
        distance 2 → cost 60
        """
        r = self.inflation_grid_radius
        for obs_y, obs_x in obstacles_index:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx = obs_x + dx
                    ny = obs_y + dy
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= r:
                        cost = int(max(0, 100 - (20 * dist)))
                        if (0 <= ny < self.grid_size and
                                0 <= nx < self.grid_size):
                            costmap[ny, nx] = max(
                                cost, costmap[ny, nx])
        return costmap

    # ──────────────────────────────────────────────────────────────
    def publish_costmap(self, grid):
        """Publish the costmap as OccupancyGrid message."""
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id  # ✅ FIX: configurable frame

        msg.info.resolution = self.grid_resolution
        msg.info.width      = self.grid_size
        msg.info.height     = self.grid_size

        # Origin = bottom-left corner of grid
        # Center grid on robot position
        msg.info.origin = Pose()
        msg.info.origin.position.x  = -self.map_center
        msg.info.origin.position.y  = -self.map_center
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten().astype(np.int8).tolist()
        self.costmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityCostmap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()