#!/usr/bin/env python3

from collections import deque
import math
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.declare_parameter('clearance_m', 0.30)
        self.declare_parameter('coverage_spacing_m', 0.85)

        self.grid = None
        self.info = None
        self.busy = False
        self.done = False
        self.mode = 'frontier'
        self.quiet_checks = 0
        self.attempts = []       # (world_x, world_y, unknown_cells_at_attempt)
        self.visited = []        # positions the robot actually passed
        self.coverage_start_unknown = None
        self.goal_handle = None
        self.cancel_requested = False
        self.best_distance = float('inf')
        self.last_progress = time.monotonic()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.timer = self.create_timer(2.0, self.tick)

    def map_callback(self, msg):
        self.grid = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        self.info = msg.info

    def robot_xy(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            p = transform.transform.translation
            return (p.x, p.y)
        except TransformException:
            return None

    def world(self, x, y):
        return (
            self.info.origin.position.x + (x + 0.5) * self.info.resolution,
            self.info.origin.position.y + (y + 0.5) * self.info.resolution,
        )

    def cell(self, xy):
        return (
            int((xy[0] - self.info.origin.position.x) / self.info.resolution),
            int((xy[1] - self.info.origin.position.y) / self.info.resolution),
        )

    @staticmethod
    def inflate(blocked, radius):
        """Mark cells within radius grid cells of a blocked cell."""
        h, w = blocked.shape
        result = np.zeros_like(blocked)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                destination_y = slice(max(0, dy), min(h, h + dy))
                destination_x = slice(max(0, dx), min(w, w + dx))
                source_y = slice(max(0, -dy), min(h, h - dy))
                source_x = slice(max(0, -dx), min(w, w - dx))
                result[destination_y, destination_x] |= blocked[
                    source_y, source_x]
        return result

    def reachable(self, safe, robot):
        h, w = safe.shape
        distances = np.full((h, w), -1, dtype=np.int32)
        rx, ry = self.cell(robot)
        if not (0 <= rx < w and 0 <= ry < h):
            return distances

        # Mapping noise can mark the cell under the robot as unsafe.
        if not safe[ry, rx]:
            ys, xs = np.nonzero(safe)
            if not len(xs):
                return distances
            nearest = np.argmin((xs - rx) ** 2 + (ys - ry) ** 2)
            if math.hypot(int(xs[nearest]) - rx, int(ys[nearest]) - ry) * \
                    self.info.resolution > 0.65:
                return distances
            rx, ry = int(xs[nearest]), int(ys[nearest])

        queue = deque([(rx, ry)])
        distances[ry, rx] = 0
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y),
                           (x, y - 1), (x, y + 1)):
                if (0 <= nx < w and 0 <= ny < h
                        and safe[ny, nx] and distances[ny, nx] < 0):
                    distances[ny, nx] = distances[y, x] + 1
                    queue.append((nx, ny))
        return distances

    def frontier_groups(self):
        free = (self.grid >= 0) & (self.grid <= 20)
        unknown = np.pad(self.grid == -1, 1, constant_values=False)
        h, w = free.shape
        edge = np.zeros_like(free)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            edge |= unknown[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        edge &= free

        groups = []
        seen = np.zeros_like(edge)
        for y, x in zip(*np.nonzero(edge)):
            if seen[y, x]:
                continue
            queue = deque([(int(x), int(y))])
            seen[y, x] = True
            group = []
            while queue:
                px, py = queue.popleft()
                group.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py),
                               (px, py - 1), (px, py + 1),
                               (px - 1, py - 1), (px + 1, py - 1),
                               (px - 1, py + 1), (px + 1, py + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and edge[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            if len(group) >= 3:
                groups.append(group)
        return groups

    def already_tried(self, xy, unknown_count, spacing):
        return any(
            math.hypot(xy[0] - x, xy[1] - y) < spacing
            and previous_unknown - unknown_count < 100
            for x, y, previous_unknown in self.attempts
        )

    def choose_frontier(self, distances, robot, unknown_count):
        h, w = distances.shape
        radius = max(1, math.ceil(0.45 / self.info.resolution))
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]

        best = None
        best_score = float('inf')
        groups = self.frontier_groups()

        for group in groups:
            candidate = None
            candidate_distance = float('inf')
            # Cap the work for a very long frontier edge.
            sample_step = max(1, len(group) // 120)
            for x, y in group[::sample_step]:
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    steps = distances[ny, nx]
                    if steps < 0 or steps >= candidate_distance:
                        continue
                    xy = self.world(nx, ny)
                    if math.dist(xy, robot) < 0.55:
                        continue
                    if self.already_tried(xy, unknown_count, 0.80):
                        continue
                    candidate = xy
                    candidate_distance = int(steps)

            if candidate is not None:
                score = (candidate_distance * self.info.resolution
                         - 0.01 * min(len(group), 100))
                if score < best_score:
                    best, best_score = candidate, score

        return best, len(groups)

    def choose_coverage(self, distances, robot, unknown_count):
        h, w = distances.shape
        spacing = self.get_parameter(
            'coverage_spacing_m').value
        stride = max(1, round(spacing / self.info.resolution))
        options = []

        for y in range(stride // 2, h, stride):
            for x in range(stride // 2, w, stride):
                if distances[y, x] < 0:
                    continue
                xy = self.world(x, y)
                if any(math.dist(xy, passed) < 0.60
                       for passed in self.visited):
                    continue
                if self.already_tried(xy, unknown_count, 0.65):
                    continue
                options.append((distances[y, x], xy))

        return min(options, default=(None, None))[1]

    def tick(self):
        if self.done or self.grid is None:
            return

        robot = self.robot_xy()
        if robot is None:
            return

        if not self.visited or math.dist(robot, self.visited[-1]) >= 0.20:
            self.visited.append(robot)

        if self.busy:
            distance = math.dist(robot, self.target)
            if distance < self.best_distance - 0.10:
                self.best_distance = distance
                self.last_progress = time.monotonic()
            if (self.goal_handle is not None and not self.cancel_requested
                    and time.monotonic() - self.last_progress > 45.0):
                self.cancel_requested = True
                self.get_logger().warn('No progress for 45 s; cancelling goal')
                self.goal_handle.cancel_goal_async()
            return

        if not self.nav.wait_for_server(timeout_sec=0.0):
            return

        clearance = self.get_parameter('clearance_m').value
        occupied = self.grid > 20
        free = (self.grid >= 0) & (self.grid <= 20)
        safe = free & ~self.inflate(
            occupied,
            max(1, math.ceil(clearance / self.info.resolution)))
        distances = self.reachable(safe, robot)
        unknown_count = int(np.count_nonzero(self.grid == -1))

        if self.mode == 'coverage':
            if (self.coverage_start_unknown is not None
                    and self.coverage_start_unknown - unknown_count >= 100):
                self.mode = 'frontier'
                self.quiet_checks = 0
                self.get_logger().info('Map grew; rechecking frontiers')

        if self.mode == 'frontier':
            target, group_count = self.choose_frontier(
                distances, robot, unknown_count)
            if target is None:
                self.quiet_checks += 1
                if self.quiet_checks < 3:
                    return
                self.mode = 'coverage'
                self.coverage_start_unknown = unknown_count
                self.get_logger().info(
                    f'No reachable new frontier; {group_count} frontier '
                    'regions remain on the map. Starting floor coverage.')
                return
            self.quiet_checks = 0
        else:
            target = self.choose_coverage(
                distances, robot, unknown_count)
            if target is None:
                self.done = True
                self.get_logger().info(
                    'DONE: no reachable unvisited coverage samples remain.')
                return

        self.target = target
        self.best_distance = math.dist(robot, target)
        self.last_progress = time.monotonic()
        self.cancel_requested = False
        self.busy = True
        self.goal_handle = None
        self.attempts.append((*target, unknown_count))

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = target
        pose.pose.orientation.w = 1.0

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(
            f'{self.mode.upper()}: goal ({target[0]:.2f}, '
            f'{target[1]:.2f})')
        self.nav.send_goal_async(goal).add_done_callback(
            self.goal_response)

    def goal_response(self, future):
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                self.get_logger().warn('Goal rejected')
                self.busy = False
                self.goal_handle = None
                return
            self.goal_handle.get_result_async().add_done_callback(
                self.goal_result)
        except Exception as exc:
            self.get_logger().error(f'Goal response error: {exc}')
            self.busy = False
            self.goal_handle = None

    def goal_result(self, future):
        try:
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Goal reached')
            else:
                self.get_logger().warn(f'Goal ended with status {status}')
        except Exception as exc:
            self.get_logger().error(f'Goal result error: {exc}')
        finally:
            self.busy = False
            self.goal_handle = None


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
