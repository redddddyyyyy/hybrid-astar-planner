"""
ROS2 Hybrid A* Planner Node.

Subscribes to:
    - /goal_pose (geometry_msgs/PoseStamped): Goal position and orientation
    - /initialpose (geometry_msgs/PoseWithCovarianceStamped): Start position
    - /map (nav_msgs/OccupancyGrid): Costmap for obstacle avoidance
    - /dynamic_obstacles (visualization_msgs/MarkerArray): Dynamic obstacles for replanning

Publishes:
    - /planned_path (nav_msgs/Path): Planned path as sequence of poses
    - /path_markers (visualization_msgs/MarkerArray): Path visualization for RViz
    - /vehicle_footprint (visualization_msgs/Marker): Vehicle footprint visualization
"""

from __future__ import annotations

import math
import sys
import os

# Add parent src to path for standalone planner imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header

# Import standalone planner
try:
    from src.hybrid_astar import HybridAStar, PlannerConfig, smooth_path
    from src.vehicle import Vehicle, VehicleConfig, State
    from src.grid import OccupancyGrid as PlannerGrid, GridConfig
except ImportError:
    # Fallback for when running as ROS2 package
    from .core.hybrid_astar import HybridAStar, PlannerConfig, smooth_path
    from .core.vehicle import Vehicle, VehicleConfig, State
    from .core.grid import OccupancyGrid as PlannerGrid, GridConfig


class HybridAStarPlannerNode(Node):
    """ROS2 node for Hybrid A* path planning."""

    def __init__(self):
        super().__init__('hybrid_astar_planner')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('vehicle.wheelbase', 2.5),
                ('vehicle.width', 1.8),
                ('vehicle.length', 4.5),
                ('vehicle.max_steer', 0.6),
                ('planner.xy_resolution', 1.0),
                ('planner.theta_resolution', 0.1),
                ('planner.step_size', 1.5),
                ('planner.include_reverse', True),
                ('planner.smooth_path', True),
                ('planner.smooth_iterations', 100),
                ('grid.resolution', 0.25),
                ('replan_on_obstacle_change', True),
                ('publish_rate', 10.0),
            ]
        )

        # Load parameters
        self._load_parameters()

        # State
        self.start_state: State | None = None
        self.goal_state: State | None = None
        self.current_path: list[State] | None = None
        self.map_received = False
        self.planner_grid: PlannerGrid | None = None
        self.dynamic_obstacles: list[tuple[float, float, float]] = []

        # QoS for map (latched)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_callback,
            10
        )
        self.start_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._start_callback,
            10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self._map_callback,
            map_qos
        )
        self.obstacle_sub = self.create_subscription(
            MarkerArray,
            '/dynamic_obstacles',
            self._obstacle_callback,
            10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        self.footprint_pub = self.create_publisher(Marker, '/vehicle_footprint', 10)

        # Timer for periodic path publishing
        period = 1.0 / self.get_parameter('publish_rate').value
        self.timer = self.create_timer(period, self._timer_callback)

        self.get_logger().info('Hybrid A* Planner Node initialized')
        self.get_logger().info(f'Vehicle: {self.vehicle_config.length}m x {self.vehicle_config.width}m')
        self.get_logger().info('Waiting for map and goal...')

    def _load_parameters(self):
        """Load ROS parameters into config objects."""
        self.vehicle_config = VehicleConfig(
            wheelbase=self.get_parameter('vehicle.wheelbase').value,
            width=self.get_parameter('vehicle.width').value,
            length=self.get_parameter('vehicle.length').value,
            max_steer=self.get_parameter('vehicle.max_steer').value,
        )

        self.planner_config = PlannerConfig(
            xy_resolution=self.get_parameter('planner.xy_resolution').value,
            theta_resolution=self.get_parameter('planner.theta_resolution').value,
            step_size=self.get_parameter('planner.step_size').value,
            include_reverse=self.get_parameter('planner.include_reverse').value,
        )

        self.vehicle = Vehicle(self.vehicle_config)

    def _goal_callback(self, msg: PoseStamped):
        """Handle new goal pose."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        theta = self._quaternion_to_yaw(msg.pose.orientation)

        self.goal_state = State(x, y, theta)
        self.get_logger().info(f'Goal received: ({x:.2f}, {y:.2f}, {theta:.2f})')

        self._plan_path()

    def _start_callback(self, msg: PoseWithCovarianceStamped):
        """Handle new start pose."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self._quaternion_to_yaw(msg.pose.pose.orientation)

        self.start_state = State(x, y, theta)
        self.get_logger().info(f'Start received: ({x:.2f}, {y:.2f}, {theta:.2f})')

        if self.goal_state is not None:
            self._plan_path()

    def _map_callback(self, msg: OccupancyGrid):
        """Handle occupancy grid map."""
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}')

        # Convert ROS OccupancyGrid to planner grid
        width = msg.info.width * msg.info.resolution
        height = msg.info.height * msg.info.resolution

        grid_config = GridConfig(
            width=width,
            height=height,
            resolution=self.get_parameter('grid.resolution').value
        )
        self.planner_grid = PlannerGrid(grid_config)

        # Copy occupancy data
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # Resample to planner resolution
        for gy in range(self.planner_grid.config.y_cells):
            for gx in range(self.planner_grid.config.x_cells):
                wx, wy = self.planner_grid.grid_to_world(gx, gy)

                # Map to source grid
                src_gx = int((wx - msg.info.origin.position.x) / msg.info.resolution)
                src_gy = int((wy - msg.info.origin.position.y) / msg.info.resolution)

                if 0 <= src_gx < msg.info.width and 0 <= src_gy < msg.info.height:
                    # Occupied if value > 50 or unknown (-1)
                    if data[src_gy, src_gx] > 50 or data[src_gy, src_gx] < 0:
                        self.planner_grid.grid[gy, gx] = True

        self.map_received = True
        self.get_logger().info('Map converted to planner grid')

    def _obstacle_callback(self, msg: MarkerArray):
        """Handle dynamic obstacle updates."""
        new_obstacles = []

        for marker in msg.markers:
            if marker.action == Marker.ADD:
                # Extract obstacle as (x, y, radius)
                x = marker.pose.position.x
                y = marker.pose.position.y
                radius = max(marker.scale.x, marker.scale.y) / 2.0
                new_obstacles.append((x, y, radius))

        if new_obstacles != self.dynamic_obstacles:
            self.dynamic_obstacles = new_obstacles
            self.get_logger().info(f'Dynamic obstacles updated: {len(new_obstacles)}')

            # Replan if enabled
            if self.get_parameter('replan_on_obstacle_change').value:
                self._plan_path()

    def _plan_path(self):
        """Execute path planning."""
        if not self.map_received:
            self.get_logger().warn('Cannot plan: map not received')
            return

        if self.start_state is None or self.goal_state is None:
            self.get_logger().warn('Cannot plan: start or goal not set')
            return

        # Create fresh grid with dynamic obstacles
        grid = self._create_planning_grid()

        # Create planner
        planner = HybridAStar(grid, self.vehicle, self.planner_config)

        self.get_logger().info('Planning path...')
        path = planner.plan(self.start_state, self.goal_state)

        if path is None:
            self.get_logger().error('Failed to find path!')
            self.current_path = None
            return

        # Smooth if enabled
        if self.get_parameter('planner.smooth_path').value:
            iterations = self.get_parameter('planner.smooth_iterations').value
            path = smooth_path(path, iterations=iterations)

        self.current_path = path
        self.get_logger().info(f'Path found: {len(path)} waypoints')

        # Publish immediately
        self._publish_path()

    def _create_planning_grid(self) -> PlannerGrid:
        """Create planning grid with dynamic obstacles."""
        # Copy base grid
        grid = PlannerGrid(self.planner_grid.config)
        grid.grid = self.planner_grid.grid.copy()

        # Add dynamic obstacles
        for x, y, radius in self.dynamic_obstacles:
            grid.add_circle(x, y, radius)

        return grid

    def _timer_callback(self):
        """Periodic callback for visualization updates."""
        if self.current_path is not None:
            self._publish_path()
            self._publish_markers()

    def _publish_path(self):
        """Publish path as nav_msgs/Path."""
        if self.current_path is None:
            return

        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for state in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = state.x
            pose.pose.position.y = state.y
            pose.pose.position.z = 0.0
            pose.pose.orientation = self._yaw_to_quaternion(state.theta)
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def _publish_markers(self):
        """Publish visualization markers."""
        markers = MarkerArray()

        # Path line
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'path'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)

        for state in self.current_path:
            p = Point()
            p.x = state.x
            p.y = state.y
            p.z = 0.1
            line_marker.points.append(p)

        markers.markers.append(line_marker)

        # Waypoint arrows (every 5th point)
        for i, state in enumerate(self.current_path[::5]):
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = 'waypoints'
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = state.x
            arrow.pose.position.y = state.y
            arrow.pose.position.z = 0.1
            arrow.pose.orientation = self._yaw_to_quaternion(state.theta)
            arrow.scale.x = 0.5
            arrow.scale.y = 0.1
            arrow.scale.z = 0.1
            arrow.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
            markers.markers.append(arrow)

        self.marker_pub.publish(markers)

        # Vehicle footprint at start
        if self.start_state:
            self._publish_footprint(self.start_state)

    def _publish_footprint(self, state: State):
        """Publish vehicle footprint."""
        footprint = self.vehicle.get_footprint(state)

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'footprint'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)

        for corner in footprint:
            p = Point()
            p.x = corner[0]
            p.y = corner[1]
            p.z = 0.05
            marker.points.append(p)

        # Close the polygon
        marker.points.append(marker.points[0])

        self.footprint_pub.publish(marker)

    @staticmethod
    def _quaternion_to_yaw(q) -> float:
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quaternion(yaw: float):
        """Convert yaw to quaternion."""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
