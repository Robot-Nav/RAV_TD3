#!/usr/bin/env python3
"""
Gazebo 动态障碍物移动控制模块
================================
控制场景中多个拖拉机模型在指定范围内来回匀速移动，
用于模拟动态障碍物环境。
"""

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates

class Moving():
    """
    动态障碍物移动管理器
    
    控制多个目标模型在起始位置和目标位置之间来回移动。
    """
    
    def __init__(self):
        """初始化移动管理器"""
        # Gazebo 模型状态发布器（用于设置模型位置）
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        
        # 需要控制的模型名称列表
        self.model_names = [
            'moving_obstacle_1',
            'moving_obstacle_2',
        ]

        # ========== 参数设置 ==========
        self.move_distance = rospy.get_param('~move_distance', 6.5)  # 移动距离（5米）
        self.publish_rate = rospy.get_param('~rate', 15.0)           # 控制频率（Hz）
        
        # 为每个模型设置不同的移动速度 (m/s)
        self.speeds = {
            'moving_obstacle_1': 0.8,
            'moving_obstacle_2': 1.2,
        }

        # ========== 状态存储 ==========
        self.start_positions = {}  # 存储每个模型的起始 X 坐标
        self.directions = {}       # 存储当前运动方向：-1 表示向左，1 表示向右
        self.target_positions = {} # 存储当前目标 X 坐标
        
        # 等待场景加载完成后获取初始位置
        rospy.sleep(2.0)
        self.get_initial_positions()
        
        # 启动移动循环
        self.moving()

    def get_initial_positions(self):
        """
        获取所有目标模型的初始位置
        
        从 Gazebo model_states 话题读取模型位置，
        并为每个模型初始化运动方向（默认向左）和目标位置。
        """
        try:
            model = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=5.0)
            name_to_idx = {name: idx for idx, name in enumerate(model.name)}
            
            for model_name in self.model_names:
                if model_name in name_to_idx:
                    idx = name_to_idx[model_name]
                    start_x = model.pose[idx].position.x
                    
                    # 记录起始位置
                    self.start_positions[model_name] = start_x
                    # 初始都向左移动（负方向）
                    self.directions[model_name] = -1.0
                    # 计算目标位置：向左移动指定距离
                    self.target_positions[model_name] = start_x - self.move_distance
                    
                else:
                    rospy.logwarn(f"{model_name} not found in Gazebo")
                    
        except rospy.ROSException:
            rospy.logerr("Failed to get initial positions")

    def moving(self):
        """
        移动控制主循环
        
        持续读取模型当前位置，判断是否需要转向，
        并发布新的模型位置到 Gazebo。
        """
        rate = rospy.Rate(self.publish_rate)
        dt = 1.0 / self.publish_rate if self.publish_rate > 0 else 0.05

        while not rospy.is_shutdown():
            # 等待 Gazebo 模型状态消息
            try:
                model = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=2.0)
            except rospy.ROSException:
                rospy.logwarn_throttle(5.0, 'Waiting for /gazebo/model_states...')
                rate.sleep()
                continue

            name_to_idx = {name: idx for idx, name in enumerate(model.name)}

            for model_name in self.model_names:
                # 检查模型是否存在
                if model_name not in name_to_idx:
                    continue
                
                # 检查是否已初始化
                if model_name not in self.start_positions:
                    continue

                idx = name_to_idx[model_name]
                curr_pose = model.pose[idx]
                curr_x = curr_pose.position.x
                
                # 获取当前状态
                direction = self.directions[model_name]
                target_x = self.target_positions[model_name]
                start_x = self.start_positions[model_name]
                speed = self.speeds[model_name]

                # 调试信息（每2秒打印一次）
                rospy.loginfo_throttle(2.0, f"{model_name}: x={curr_x:.2f}, dir={direction}, target={target_x:.2f}, start={start_x:.2f}")

                # --- 转向逻辑 ---
                # 到达最左端 → 改为向右移动
                if direction < 0 and curr_x <= target_x:
                    rospy.loginfo(f"{model_name} reached LEFT target, turning RIGHT")
                    self.directions[model_name] = 1.0        # 改为向右
                    self.target_positions[model_name] = start_x  # 目标改回起始位置
                
                # 回到起始位置 → 改为向左移动
                elif direction > 0 and curr_x >= start_x:
                    rospy.loginfo(f"{model_name} reached START position, turning LEFT")
                    self.directions[model_name] = -1.0           # 改为向左
                    self.target_positions[model_name] = start_x - self.move_distance  # 目标改回左边指定距离处

                # --- 位置更新 ---
                # 使用运动学公式计算下一帧位置（更稳定）
                cmd = ModelState()
                cmd.model_name = model_name
                cmd.pose = curr_pose
                cmd.pose.orientation.x = 0.0
                cmd.pose.orientation.y = 0.0
                cmd.pose.orientation.z = 0.0
                cmd.pose.orientation.w = 1.0

                # 计算下一个位置：x = x + direction * speed * dt
                next_x = curr_x + self.directions[model_name] * speed * dt
                # 边界限制：不超出目标范围
                if self.directions[model_name] < 0:
                    next_x = max(target_x, next_x)  # 向左不超出目标
                else:
                    next_x = min(start_x, next_x)   # 向右不超出起始位置
                cmd.pose.position.x = next_x

                cmd.twist = Twist()
                cmd.twist.linear.x = 0.0
                cmd.reference_frame = 'world'

                self.pub_model.publish(cmd)

            rate.sleep()


def main():
    """节点入口函数"""
    rospy.init_node('moving_1')
    moving = Moving()


if __name__ == '__main__':
    main()
