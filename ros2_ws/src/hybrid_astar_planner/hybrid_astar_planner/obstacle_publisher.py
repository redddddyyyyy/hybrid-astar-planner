"""
Dynamic obstacle publisher for testing replanning.

Publishes moving obstacles as MarkerArray to trigger path replanning.
"""

import math
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class DynamicObstaclePublisher(Node):
    """Publishes dynamic obstacles for testing replanning."""

    def __init__(self):
        super().__init__('obstacle_publisher')

        # Parameters
        self.declare_parameter('num_obstacles', 3)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('speed', 0.5)

        self.num_obstacles = self.get_parameter('num_obstacles').value
        self.speed = self.get_parameter('speed').value

        # Publisher
        self.publisher = self.create_publisher(
            MarkerArray,
            '/dynamic_obstacles',
            10
        )

        # Timer
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)

        # Obstacle state: [(x, y, radius, vx, vy), ...]
        self.obstacles = []
        self._init_obstacles()

        self.get_logger().info(f'Publishing {self.num_obstacles} dynamic obstacles')

    def _init_obstacles(self):
        """Initialize obstacle positions and velocities."""
        import random

        for i in range(self.num_obstacles):
            x = random.uniform(10, 40)
            y = random.uniform(10, 40)
            radius = random.uniform(1.0, 2.0)
            angle = random.uniform(0, 2 * math.pi)
            vx = self.speed * math.cos(angle)
            vy = self.speed * math.sin(angle)
            self.obstacles.append([x, y, radius, vx, vy])

    def _timer_callback(self):
        """Update and publish obstacles."""
        dt = 0.1  # Time step

        markers = MarkerArray()

        for i, obs in enumerate(self.obstacles):
            # Update position
            obs[0] += obs[3] * dt
            obs[1] += obs[4] * dt

            # Bounce off boundaries
            if obs[0] < 5 or obs[0] > 45:
                obs[3] *= -1
            if obs[1] < 5 or obs[1] > 45:
                obs[4] *= -1

            # Create marker
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'dynamic_obstacles'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = obs[0]
            marker.pose.position.y = obs[1]
            marker.pose.position.z = 0.5
            marker.scale.x = obs[2] * 2
            marker.scale.y = obs[2] * 2
            marker.scale.z = 1.0
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
            markers.markers.append(marker)

        self.publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstaclePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
