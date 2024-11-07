import sys
import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from tf_transformations import quaternion_from_euler
from rclpy.action import ActionClient

class StaticGoalSender(Node):
    def __init__(self):
        super().__init__('static_goal_sender')
        self.move_action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info('MoveGroup action client initialized.')

    def send_static_goal(self):
        # Wait for the action server to be available
        if not self.move_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available. Ensure that MoveIt is running.')
            return

        # Create the MoveGroup goal
        goal_msg = MoveGroup.Goal()

        # Assign group_name and end_effector_name
        goal_msg.request.group_name = 'panda_arm'          # Adjust if different
        goal_msg.request.end_effector_name = 'panda_hand'  # Adjust if different

        # Populate the MotionPlanRequest
        goal_msg.request.motion_plan_request.group_name = 'panda_arm'
        goal_msg.request.motion_plan_request.num_planning_attempts = 5
        goal_msg.request.motion_plan_request.allowed_planning_time = 5.0  # seconds
        goal_msg.request.motion_plan_request.workspace_parameters.header.frame_id = 'panda_link0'

        # Define goal constraints
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive

        constraints = Constraints()

        # Position Constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = 'panda_link0'
        position_constraint.link_name = 'panda_hand'
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Define the bounding volume for position constraint
        bounding_volume = SolidPrimitive()
        bounding_volume.type = SolidPrimitive.BOX
        bounding_volume.dimensions = [0.05, 0.05, 0.05]  # 5cm cube

        position_constraint.constraint_region.primitives.append(bounding_volume)
        position_constraint.constraint_region.primitive_poses.append(self.create_target_pose())
        position_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)

        # Orientation Constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = 'panda_link0'
        orientation_constraint.link_name = 'panda_hand'
        q = quaternion_from_euler(0, 0, 0)  # Neutral orientation
        orientation_constraint.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        orientation_constraint.absolute_x_axis_tolerance = 0.2
        orientation_constraint.absolute_y_axis_tolerance = 0.2
        orientation_constraint.absolute_z_axis_tolerance = 0.2
        orientation_constraint.weight = 1.0

        constraints.orientation_constraints.append(orientation_constraint)

        # Assign constraints to the goal
        goal_msg.request.motion_plan_request.goal_constraints.append(constraints)

        # Set velocity and acceleration scaling factors
        goal_msg.request.motion_plan_request.max_velocity_scaling_factor = 0.1
        goal_msg.request.motion_plan_request.max_acceleration_scaling_factor = 0.1

        # Send the goal
        self.get_logger().info('Sending static goal to MoveGroup action server...')
        send_goal_future = self.move_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def create_target_pose(self):
        pose = Pose()
        pose.position.x = 0.4
        pose.position.y = 0.0
        pose.position.z = 0.5
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        return pose

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by MoveGroup action server')
            return

        self.get_logger().info('Goal accepted by MoveGroup action server')
        self.get_logger().info('Waiting for result...')

        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        error_code = result.error_code.val

        if error_code == MoveGroup.Result.SUCCESS:
            self.get_logger().info('MoveGroup action succeeded!')
        else:
            self.get_logger().error(f'MoveGroup action failed with error code: {error_code}')

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = StaticGoalSender()
    node.send_static_goal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
