import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import PoseStamped, Quaternion
from tf_transformations import quaternion_from_euler

class StaticGoalSender(Node):
    def __init__(self):
        super().__init__('static_goal_sender')
        self._action_client = ActionClient(self, MoveGroup, 'move_group')

    def send_goal(self):
        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()

        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = 'panda_link0'

        # Define a target pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'panda_link0'
        target_pose.pose.position.x = 0.4
        target_pose.pose.position.y = 0.0
        target_pose.pose.position.z = 0.5
        q = quaternion_from_euler(0, 0, 0)
        target_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Create constraints
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive

        constraints = Constraints()

        # Position constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = 'panda_link0'
        position_constraint.link_name = 'panda_hand'
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Define the bounding volume for position constraint
        bounding_volume = SolidPrimitive()
        bounding_volume.type = SolidPrimitive.BOX
        bounding_volume.dimensions = [0.01, 0.01, 0.01]  # Small box around the target

        position_constraint.constraint_region.primitives.append(bounding_volume)
        position_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        position_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)

        # Orientation constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = 'panda_link0'
        orientation_constraint.link_name = 'panda_hand'
        orientation_constraint.orientation = target_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        constraints.orientation_constraints.append(orientation_constraint)

        goal_msg.request.goal_constraints.append(constraints)
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = 2.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        self.get_logger().info('Sending goal to MoveGroup...')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by MoveGroup action server')
            return

        self.get_logger().info('Goal accepted by MoveGroup action server')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info('MoveGroup action succeeded!')
        else:
            self.get_logger().error(f'MoveGroup action failed with error code: {result.error_code.val}')

def main(args=None):
    rclpy.init(args=args)
    node = StaticGoalSender()
    node.send_goal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
