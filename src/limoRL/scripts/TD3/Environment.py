#!/usr/bin/python3
"""
TD3 强化学习环境模块
=====================
本模块实现了基于 Gazebo 仿真的移动机器人导航环境，包含：
- 激光雷达传感器数据获取与预处理
- 里程计位姿订阅与目标角度计算
- 多目标奖励函数设计（目标推进、朝向对齐、安全间隙、控制平滑等）
- 碰撞检测、旋转卡死检测、无进度卡死检测
- 速度限幅与安全减速策略
"""

import rospy
import numpy as np
import math
import copy
import time
from math import pi
from .respawnGoal import Respawn
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Empty
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class Env():
    """
    强化学习环境类
    
    封装 Gazebo 仿真交互接口，提供 Gym 风格的 reset/step 方法。
    状态空间：激光雷达降采样数据 + 目标朝向角 + 目标距离
    动作空间：[线速度, 角速度]
    """
    
    def __init__(self, action_dim=2):
        """
        初始化环境
        
        Args:
            action_dim: 动作维度，默认为 2（线速度 + 角速度）
        """
        # ========== 目标状态相关 ==========
        self.goal_x = 0                # 目标点 x 坐标
        self.goal_y = 0                # 目标点 y 坐标
        self.heading = 0               # 机器人朝向与目标方向的夹角（弧度）
        self.initGoal = True           # 首次初始化标志
        self.get_goalbox = False       # 是否到达目标区域标志
        
        # ========== 传感器数据缓存 ==========
        self.position = Pose()         # 机器人当前位置
        self.scan = None               # 最新激光雷达扫描数据
        self.scan_seq = 0              # 激光雷达数据序列号（用于检测新帧）
        self.scan_topic = rospy.get_param('~scan_topic', '/limo/scan')          # 激光雷达主话题
        self.scan_fallback_topics = rospy.get_param('~scan_fallback_topics', ['/scan'])  # 备用话题列表
        self.scan_wait_timeout = float(rospy.get_param('~scan_wait_timeout', 20.0))      # 等待激光数据超时时间
        
        # 确保备用话题为列表格式
        if isinstance(self.scan_fallback_topics, str):
            self.scan_fallback_topics = [self.scan_fallback_topics]
            
        # ========== ROS 通信接口 ==========
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)      # 速度指令发布器
        self.pub_path = rospy.Publisher('limo_rl/path', Path, queue_size=10)    # 机器人轨迹
        self.pub_goal = rospy.Publisher('limo_rl/goal', PoseStamped, queue_size=5)  # 目标点
        self.pub_risk_markers = rospy.Publisher('limo_rl/risk_markers', MarkerArray, queue_size=5)  # 风险区域
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)     # Gazebo 世界重置服务
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)  # 物理引擎恢复服务
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)      # 物理引擎暂停服务
        self.respawn_goal = Respawn()    # 目标点重新生成管理器
        
        # ========== 距离跟踪变量 ==========
        self.last_distance = 0         # 上一步到目标的距离
        self.past_distance = 0.        # 历史参考距离（用于计算奖励）
        self.initial_diatance = 0.     # 初始距离（episode 开始时）
        
        # ========== 异常状态计数器 ==========
        self.stopped = 0               # 停止状态计数
        self.spin_streak = 0           # 连续原地旋转步数（用于检测卡死）
        self.no_progress_streak = 0    # 无进展连续步数（用于检测卡死）
        self.reset_count = 0           # 累计 reset 次数
        
        # ========== 系统参数 ==========
        self.action_dim = action_dim   # 动作空间维度
        self.odom_valid = True         # 里程计数据有效性标志
        self.last_cmd_linear = 0.0     # 上一次线速度指令
        self.last_cmd_angular = 0.0    # 上一次角速度指令

        # ============================================================
        # 可配置的环境阈值参数（支持通过 ROS 参数服务器动态调整）
        # ============================================================
        
        # --- 传感器处理参数 ---
        self.scan_clip = rospy.get_param('~scan_clip', 6.0)                    # 激光雷达最大截断距离
        self.scan_num_sectors = max(16, int(rospy.get_param('~scan_num_sectors', 36)))  # 降采样后的扇区数量
        self.front_sector_half_width = max(1, int(rospy.get_param('~front_sector_half_width', 3)))  # 前方扇区半宽
        self.min_valid_scan = float(rospy.get_param('~min_valid_scan', 0.16))  # 最小有效扫描距离

        # --- 碰撞与安全参数 ---
        self.collision_range = rospy.get_param('~collision_range', 0.20)       # 碰撞判定距离
        self.emergency_stop_range = float(rospy.get_param('~emergency_stop_range', 0.16))  # 紧急停止距离
        self.slowdown_range_1 = float(rospy.get_param('~slowdown_range_1', 0.24))          # 一级减速距离
        self.slowdown_speed_1 = float(rospy.get_param('~slowdown_speed_1', 0.08))          # 一级减速限速值
        self.slowdown_range_2 = float(rospy.get_param('~slowdown_range_2', 0.45))          # 二级减速距离
        self.slowdown_speed_2 = float(rospy.get_param('~slowdown_speed_2', 0.22))          # 二级减速限速值
        self.collision_risk_range = float(rospy.get_param('~collision_risk_range', 0.85))  # 碰撞风险判定距离

        # --- 目标到达参数 ---
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.35)         # 目标到达容差

        # --- 进度检测参数 ---
        self.progress_epsilon = rospy.get_param('~progress_epsilon', 0.01)     # 有效进展最小阈值
        self.max_no_progress_steps = rospy.get_param('~max_no_progress_steps', 200)  # 最大无进展步数

        # --- 奖励缩放系数 ---
        self.goal_reward_scale = float(rospy.get_param('~goal_reward_scale', 22.0))        # 目标推进奖励系数
        self.heading_reward_scale = float(rospy.get_param('~heading_reward_scale', 1.6))   # 朝向对齐奖励系数
        self.clearance_reward_scale = float(rospy.get_param('~clearance_reward_scale', 2.0))  # 安全间隙奖励系数
        self.turn_reward_scale = float(rospy.get_param('~turn_reward_scale', 1.2))         # 转向奖励系数
        self.smooth_penalty_scale = float(rospy.get_param('~smooth_penalty_scale', 0.25))  # 控制平滑惩罚系数

        # --- 目标切换策略 ---
        self.random_goal_on_reset = rospy.get_param('~random_goal_on_reset', False)        # reset 时是否随机切换目标
        self.goal_reset_interval = max(1, int(rospy.get_param('~goal_reset_interval', 5))) # 目标切换间隔

        # --- 运动学约束参数 ---
        self.max_cmd_linear = float(rospy.get_param('~max_cmd_linear', 0.65))              # 最大线速度
        self.max_cmd_angular = float(rospy.get_param('~max_cmd_angular', 1.0))             # 最大角速度
        self.max_cmd_delta_linear = float(rospy.get_param('~max_cmd_delta_linear', 0.08))  # 线速度最大变化率
        self.max_cmd_delta_angular = float(rospy.get_param('~max_cmd_delta_angular', 0.20)) # 角速度最大变化率

        # --- 异常检测边界 ---
        self.max_valid_position_abs = float(rospy.get_param('~max_valid_position_abs', 30.0))   # 最大有效位置绝对值
        self.max_valid_goal_distance = float(rospy.get_param('~max_valid_goal_distance', 80.0)) # 最大有效目标距离

        # ========== 订阅传感器数据 ==========
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)    # 里程计回调
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.getScan)  # 激光雷达回调

        self.latest_front_min = self.scan_clip  # 最新前方最小障碍物距离

        # ========== 可视化追踪变量 ==========
        self.path_msg = Path()                    # 机器人轨迹消息
        self.path_msg.header.frame_id = 'odom'
        self.current_yaw = 0.0                    # 当前机器人朝向（用于风险Marker方向计算）
        self._risk_marker_id = 0                  # 风险Marker ID计数器

        # 注册节点关闭回调函数
        rospy.on_shutdown(self.shutdown)
        
    def shutdown(self):
        """节点关闭时发布零速度指令，确保安全停车"""
        rospy.loginfo("Stopping JZJBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)
    
    def getGoalDistace(self):
        """
        计算机器人到目标点的欧氏距离
        
        Returns:
            float: 目标距离
        """
        goal_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        self.past_distance = goal_distance
        self.initial_diatance = goal_distance
        return goal_distance
    
    def getOdometry(self, odom):
        """
        里程计回调函数，更新机器人位姿和朝向角
        
        计算逻辑：
        1. 提取位置和四元数姿态
        2. 将四元数转换为欧拉角 yaw
        3. 验证数据有效性（非NaN、非Inf、在合理范围内）
        4. 计算目标方位角与当前 yaw 的差值，归一化到 [-π, π]
        
        Args:
            odom: Odometry 消息
        """
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        # 数据有效性校验：排除 NaN/Inf
        if (not np.isfinite(self.position.x)) or (not np.isfinite(self.position.y)) or (not np.isfinite(yaw)):
            self.odom_valid = False
            return
        # 位置合理性校验：防止仿真异常导致的超大坐标
        if abs(self.position.x) > self.max_valid_position_abs or abs(self.position.y) > self.max_valid_position_abs:
            self.odom_valid = False
            return
        self.odom_valid = True
        
        # 计算目标方向角（相对于世界坐标系）
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw  # 目标方向与机器人朝向的偏差
        
        # 归一化到 [-π, π]
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)
        self.current_yaw = yaw

        # 发布机器人轨迹
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'odom'
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = self.position.x
        pose_stamped.pose.position.y = self.position.y
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = orientation
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(pose_stamped)
        if len(self.path_msg.poses) > 2000:
            self.path_msg.poses = self.path_msg.poses[-1500:]
        self.pub_path.publish(self.path_msg)

    def _sanitize_action(self, action, past_action):
        """
        动作安全过滤与限幅
        
        处理流程：
        1. NaN/Inf 值替换为 0
        2. 绝对值限幅（max_cmd_linear / max_cmd_angular）
        3. 变化率限幅（max_cmd_delta）：防止速度突变
        
        Args:
            action: 当前动作 [线速度, 角速度]
            past_action: 上一动作
            
        Returns:
            tuple: (safe_linear, safe_angular)
        """
        if action is None or len(action) < 2:
            return 0.0, 0.0

        # NaN/Inf 防护
        linear = float(action[0]) if np.isfinite(action[0]) else 0.0
        angular = float(action[1]) if np.isfinite(action[1]) else 0.0

        # 绝对值限幅
        linear = np.clip(linear, 0.0, self.max_cmd_linear)
        angular = np.clip(angular, -self.max_cmd_angular, self.max_cmd_angular)

        # 获取上一动作值
        if past_action is not None and len(past_action) >= 2:
            past_linear = float(past_action[0]) if np.isfinite(past_action[0]) else 0.0
            past_angular = float(past_action[1]) if np.isfinite(past_action[1]) else 0.0
        else:
            past_linear = self.last_cmd_linear
            past_angular = self.last_cmd_angular

        # 变化率限幅：确保速度平滑变化
        linear = np.clip(
            linear,
            past_linear - self.max_cmd_delta_linear,
            past_linear + self.max_cmd_delta_linear,
        )
        angular = np.clip(
            angular,
            past_angular - self.max_cmd_delta_angular,
            past_angular + self.max_cmd_delta_angular,
        )

        self.last_cmd_linear = float(linear)
        self.last_cmd_angular = float(angular)
        return float(linear), float(angular)

    def getScan(self, scan):
        """
        激光雷达数据回调函数，仅缓存最新数据和递增序列号
        
        Args:
            scan: LaserScan 消息
        """
        self.scan = scan
        self.scan_seq += 1

    def _retarget_scan_topic(self, new_topic):
        """
        切换激光雷达订阅话题（主话题不可用时自动切换到备用话题）
        
        Args:
            new_topic: 新的话题名称
        """
        if new_topic == self.scan_topic:
            return
        try:
            self.sub_scan.unregister()
        except Exception:
            pass
        self.scan_topic = new_topic
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.getScan)
        rospy.logwarn('Switched LaserScan topic to %s', self.scan_topic)

    def _try_scan_fallback(self, timeout_per_topic=1.5):
        """
        尝试从备用话题列表获取激光数据
        
        Args:
            timeout_per_topic: 每个话题的等待超时时间
            
        Returns:
            LaserScan: 获取到的激光数据
            
        Raises:
            rospy.ROSException: 所有话题均超时时抛出
        """
        topics = [self.scan_topic] + list(self.scan_fallback_topics)
        seen = set()
        ordered_topics = []
        # 去重并保持顺序
        for t in topics:
            if t and t not in seen:
                ordered_topics.append(t)
                seen.add(t)

        for topic in ordered_topics:
            try:
                msg = rospy.wait_for_message(topic, LaserScan, timeout=timeout_per_topic)
                self.scan = msg
                self.scan_seq += 1
                if topic != self.scan_topic:
                    self._retarget_scan_topic(topic)
                return copy.deepcopy(msg)
            except rospy.ROSException:
                continue

        raise rospy.ROSException(
            'Timed out waiting for LaserScan. Tried topics: %s. '
            'Please ensure Gazebo and robot sensors are running.' % ', '.join(ordered_topics)
        )

    def _wait_for_scan(self, timeout=5.0, require_new=False, last_seq=-1):
        """
        等待激光雷达数据
        
        Args:
            timeout: 等待超时时间
            require_new: 是否要求新帧（True 则要求 scan_seq > last_seq）
            last_seq: 上一帧序列号
            
        Returns:
            LaserScan: 激光数据副本
        """
        if timeout <= 0:
            if self.scan is not None and ((not require_new) or (self.scan_seq > last_seq)):
                return copy.deepcopy(self.scan)
            raise rospy.ROSException('No LaserScan available for immediate fallback')

        start = rospy.get_time()
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.scan is not None:
                if (not require_new) or (self.scan_seq > last_seq):
                    return copy.deepcopy(self.scan)
            if timeout > 0 and (rospy.get_time() - start) > timeout:
                return self._try_scan_fallback(timeout_per_topic=1.5)
            rate.sleep()
        return copy.deepcopy(self.scan)

    def _sanitize_scan(self, scan):
        """
        清洗激光雷达数据：处理 Inf/NaN/负值
        
        处理规则：
        - Inf 或超过最大值 → 替换为 scan_clip
        - NaN → 替换为 scan_clip
        - 负值 → 取 max(r, 0.0)
        
        Args:
            scan: 原始 LaserScan 消息
            
        Returns:
            list: 清洗后的距离列表
        """
        scan_range = []
        for i in range(len(scan.ranges)):
            r = scan.ranges[i]
            if r == float('Inf') or r > self.scan_clip:
                scan_range.append(self.scan_clip)
            elif np.isnan(r):
                scan_range.append(self.scan_clip)
            else:
                scan_range.append(max(float(r), 0.0))
        return scan_range

    def _downsample_scan(self, scan_range, num_sectors=None):
        """
        降采样激光数据：将原始扫描数据聚合为固定数量的扇区
        
        每个扇区取该区间内的最小距离值（保守策略，确保安全）
        
        Args:
            scan_range: 清洗后的激光距离列表
            num_sectors: 目标扇区数，默认使用 self.scan_num_sectors
            
        Returns:
            list: 降采样后的距离列表
        """
        if num_sectors is None:
            num_sectors = self.scan_num_sectors
        sector_size = max(1, len(scan_range) // num_sectors)
        downsampled_scan = []
        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = min(start_idx + sector_size, len(scan_range))
            if start_idx < len(scan_range):
                downsampled_scan.append(min(scan_range[start_idx:end_idx]))
            else:
                downsampled_scan.append(self.scan_clip)
        return downsampled_scan

    def _front_min_from_downsampled(self, downsampled_scan):
        """
        提取降采样数据中前方扇区的最小距离值
        
        Args:
            downsampled_scan: 降采样后的激光数据
            
        Returns:
            float: 前方最小距离
        """
        center = len(downsampled_scan) // 2
        start_idx = max(0, center - self.front_sector_half_width)
        end_idx = min(len(downsampled_scan), center + self.front_sector_half_width + 1)
        return min(downsampled_scan[start_idx:end_idx])

    def _limit_linear_speed(self, linear_cmd, front_min_range):
        """
        根据前方障碍物距离动态限制线速度
        
        减速策略（由近到远）：
        - 紧急停止距离内 → 速度 0
        - 一级减速距离内 → 限速 slowdown_speed_1
        - 二级减速距离内 → 限速 slowdown_speed_2
        - 安全距离外 → 不限速
        
        Args:
            linear_cmd: 原始线速度指令
            front_min_range: 前方最小障碍物距离
            
        Returns:
            float: 限制后的线速度
        """
        linear_cmd = max(0.0, float(linear_cmd))
        if front_min_range <= self.emergency_stop_range:
            return 0.0
        if front_min_range <= self.slowdown_range_1:
            return min(linear_cmd, self.slowdown_speed_1)
        if front_min_range <= self.slowdown_range_2:
            return min(linear_cmd, self.slowdown_speed_2)
        return linear_cmd
        
    def _publish_goal(self):
        """发布目标点 PoseStamped 消息供 RViz 可视化"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'odom'
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = self.goal_x
        goal_msg.pose.position.y = self.goal_y
        goal_msg.pose.position.z = 0.05
        goal_msg.pose.orientation.w = 1.0
        self.pub_goal.publish(goal_msg)

    def _publish_risk_markers(self, downsampled_scan):
        """
        发布风险区域 MarkerArray 供 RViz 可视化
        
        对降采样后的激光数据，将风险扇区（距离 < collision_risk_range）
        以扇形 Marker 形式展示在机器人周围，颜色从绿到红表示风险等级。
        
        Args:
            downsampled_scan: 降采样后的激光距离列表
        """
        markers = MarkerArray()
        now = rospy.Time.now()

        # 清除上一帧 Marker
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        num_sectors = len(downsampled_scan)
        angle_min = -pi
        angle_increment = 2 * pi / num_sectors
        robot_x = self.position.x
        robot_y = self.position.y
        yaw = self.current_yaw

        for i, dist in enumerate(downsampled_scan):
            if dist >= self.collision_risk_range:
                continue

            risk_val = np.clip(
                (self.collision_risk_range - dist) /
                max(1e-6, (self.collision_risk_range - self.collision_range)),
                0.0, 1.0,
            )

            angle = angle_min + (i + 0.5) * angle_increment
            world_angle = yaw + angle

            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = now
            marker.ns = 'risk_sector'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = robot_x
            marker.pose.position.y = robot_y
            marker.pose.position.z = 0.05

            arrow_len = min(dist, self.collision_risk_range) * 0.6
            end_x = robot_x + arrow_len * math.cos(world_angle)
            end_y = robot_y + arrow_len * math.sin(world_angle)

            marker.pose.orientation = Quaternion(
                *quaternion_from_euler(0, 0, world_angle)
            )

            marker.scale.x = arrow_len
            marker.scale.y = 0.04
            marker.scale.z = 0.04

            r = risk_val
            g = 1.0 - risk_val
            marker.color = ColorRGBA(r=r, g=g, b=0.0, a=0.6 + 0.4 * risk_val)

            marker.lifetime = rospy.Duration(0.3)
            markers.markers.append(marker)

        # 发布机器人前方扇区范围指示
        front_marker = Marker()
        front_marker.header.frame_id = 'odom'
        front_marker.header.stamp = now
        front_marker.ns = 'front_range'
        front_marker.id = 0
        front_marker.type = Marker.SPHERE
        front_marker.action = Marker.ADD
        front_marker.pose.position.x = robot_x + self.collision_risk_range * 0.5 * math.cos(yaw)
        front_marker.pose.position.y = robot_y + self.collision_risk_range * 0.5 * math.sin(yaw)
        front_marker.pose.position.z = 0.02
        front_marker.pose.orientation.w = 1.0
        front_marker.scale.x = self.collision_risk_range
        front_marker.scale.y = self.collision_risk_range
        front_marker.scale.z = 0.01
        front_marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.08)
        front_marker.lifetime = rospy.Duration(0.3)
        markers.markers.append(front_marker)

        self.pub_risk_markers.publish(markers)
        
    def getState(self, scan, past_action):
        """
        构建状态向量
        
        状态组成：[降采样激光数据(N维), heading(1维), distance(1维)]
        同时执行碰撞检测和到达检测
        
        Args:
            scan: LaserScan 消息
            past_action: 上一动作
            
        Returns:
            tuple: (state, done)
                - state: 状态向量
                - done: 是否结束（碰撞或异常）
        """
        heading = self.heading
        min_range = self.collision_range
        done = False

        # 里程计异常 → 返回安全状态并标记结束
        if not self.odom_valid:
            return [self.scan_clip] * self.scan_num_sectors + [0.0, self.max_valid_goal_distance], True

        # 激光数据预处理
        scan_range = self._sanitize_scan(scan)
        downsampled_scan = self._downsample_scan(scan_range)
        front_min_range = self._front_min_from_downsampled(downsampled_scan)
        self.latest_front_min = front_min_range

        # 碰撞检测：使用前方最小距离，避免尾部/车身自反射误判
        collision_eval_range = max(front_min_range, self.min_valid_scan)
        if collision_eval_range < min_range:
            rospy.logwarn(
                "Collision detected! front_min_scan: %.3f, collision_range: %.3f",
                front_min_range,
                min_range,
            )
            done = True
            
        # 计算当前到目标的距离
        current_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        # 距离异常检测
        if (not np.isfinite(current_distance)) or current_distance > self.max_valid_goal_distance:
            return [self.scan_clip] * self.scan_num_sectors + [0.0, self.max_valid_goal_distance], True
        # 到达目标检测
        if current_distance < self.goal_tolerance:
            self.get_goalbox = True

        # 返回状态向量：激光特征 + 目标极坐标
        return downsampled_scan + [heading, current_distance], done
    
    def setReward(self, state, action, past_action, done):
        """
        计算多目标奖励函数（RAV-TD3 v2 版本）
        
        改进点：
        1. 效率奖励：鼓励快速到达目标（步数越少奖励越高）
        2. 存活奖励递减：防止绕圈刷分
        3. 碰撞惩罚加重：增强PER区分度
        4. 到达目标奖励优化：与距离和步数相关
        
        奖励组成：
        1. goal_reward: 目标推进奖励
        2. heading_reward: 朝向对齐奖励
        3. clearance_reward: 安全间隙奖励
        4. speed_reward: 速度奖励
        5. turn_reward: 转向奖励
        6. smooth_penalty: 控制平滑惩罚
        7. risky_straight_penalty: 高风险下直走惩罚
        8. risky_speed_penalty: 高风险下高速惩罚
        9. efficiency_reward: 效率奖励（步数越少奖励越高）
        10. survival_reward: 递减生存奖励
        
        Args:
            state: 当前状态向量
            action: 当前动作
            past_action: 上一动作
            done: 是否因碰撞结束
            
        Returns:
            tuple: (reward, done)
        """
        collision = bool(done)

        heading = state[-2]
        current_distance = state[-1]
        front_min_range = self.latest_front_min

        # NaN/Inf 防护
        if (not np.isfinite(heading)) or (not np.isfinite(current_distance)):
            return -220.0, True

        # 计算进度（距离减小为正）
        progress = self.past_distance - current_distance
        progress = float(np.clip(progress, -0.8, 0.8))
        heading_alignment = math.cos(heading)

        # 计算碰撞风险系数 [0, 1]
        risk = np.clip(
            (self.collision_risk_range - front_min_range) /
            max(1e-6, (self.collision_risk_range - self.collision_range)),
            0.0,
            1.0,
        )

        # --- 基础奖励计算 ---
        goal_reward = self.goal_reward_scale * progress * (1.0 - 0.35 * risk)
        heading_reward = self.heading_reward_scale * heading_alignment
        clearance_reward = self.clearance_reward_scale * np.tanh((front_min_range - self.collision_range) * 3.0)
        speed_reward = 1.0 * action[0] * (1.0 - 0.5 * risk)
        turn_reward = self.turn_reward_scale * abs(action[1]) * (0.25 + risk)
        smooth_penalty = -self.smooth_penalty_scale * (
            abs(action[0] - past_action[0]) + 0.8 * abs(action[1] - past_action[1])
        )

        risky_straight_penalty = -2.5 * risk * max(0.22 - abs(action[1]), 0.0)
        risky_speed_penalty = -2.0 * risk * max(action[0] - 0.25, 0.0)

        # --- 改进1：效率奖励（距离目标越近，每步奖励越高）---
        # 当接近目标时，给予更高的正向反馈
        if current_distance < 2.0:
            efficiency_reward = 0.5 * (1.0 - current_distance / 2.0)
        else:
            efficiency_reward = 0.0

        # --- 改进2：递减生存奖励（防止绕圈刷分）---
        # 随着步数增加，生存奖励递减到0
        total_steps_in_episode = self.spin_streak + self.no_progress_streak
        survival_decay = max(0.0, 1.0 - total_steps_in_episode / 500.0)
        survival_reward = 0.05 * survival_decay

        reward = (
            goal_reward
            + heading_reward
            + clearance_reward
            + speed_reward
            + turn_reward
            + smooth_penalty
            + risky_straight_penalty
            + risky_speed_penalty
            + efficiency_reward
            + survival_reward
        )

        # --- 无进展检测 ---
        if progress > self.progress_epsilon:
            self.no_progress_streak = 0
        elif current_distance > self.goal_tolerance:
            self.no_progress_streak += 1
            reward -= 0.35

        # 近距离高速惩罚
        if front_min_range < 0.30 and action[0] > 0.20:
            reward -= 1.0

        # --- 旋转卡死检测 ---
        if action[0] < 0.05 and abs(action[1]) > 0.6:
            self.spin_streak += 1
            reward -= 0.5
        else:
            self.spin_streak = 0

        if self.spin_streak > 120:
            rospy.loginfo("Spin stuck detected")
            reward -= 80.0
            done = True

        # 无进展超过阈值
        if self.no_progress_streak > self.max_no_progress_steps:
            rospy.loginfo("No-progress stuck detected")
            reward -= 100.0
            done = True

        self.past_distance = current_distance

        # 奖励 NaN/Inf 防护
        if not np.isfinite(reward):
            reward = -200.0
            done = True

        # --- 碰撞处理（改进3：加重碰撞惩罚）---
        if collision:
            rospy.loginfo("Collision!!")
            reward = -350.  # 从 -260 提升到 -350
            self.pub_cmd_vel.publish(Twist())
            self.respawn_goal.index = 0
            self.spin_streak = 0
            self.no_progress_streak = 0

        # --- 到达目标处理（改进4：奖励与距离和效率相关）---
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            # 基础奖励 + 距离奖励（越近奖励越高）+ 效率奖励
            base_reward = 300.
            distance_bonus = max(0., 100. * (1.0 - self.goal_distance / 10.0))
            reward = base_reward + distance_bonus
            done = True
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            self.spin_streak = 0
            self.no_progress_streak = 0

        return reward, done
     
    def step(self, action, past_action):
        """
        执行一步环境交互
        
        流程：
        1. 动作安全过滤
        2. 根据前方障碍物动态限速
        3. 发布速度指令
        4. 等待新的激光数据
        5. 构建状态
        6. 计算奖励
        7. 结束 episode 时发布零速度
        
        Args:
            action: 网络输出的动作
            past_action: 上一动作
            
        Returns:
            tuple: (state, reward, done)
        """
        linear_vel, ang_vel = self._sanitize_action(action, past_action)

        # 里程计异常 → 直接返回失败状态
        if not self.odom_valid:
            self.pub_cmd_vel.publish(Twist())
            safe_state = np.asarray([self.scan_clip] * self.scan_num_sectors + [0.0, self.max_valid_goal_distance])
            return safe_state, -220.0, True

        # 根据前方障碍物动态限制线速度
        latest_downsampled = None
        if self.scan is not None:
            latest_scan = self._sanitize_scan(self.scan)
            latest_ds = self._downsample_scan(latest_scan)
            latest_front_min = self._front_min_from_downsampled(latest_ds)
            linear_vel = self._limit_linear_speed(linear_vel, latest_front_min)
            latest_downsampled = latest_ds

        # 发布速度指令
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        prev_scan_seq = self.scan_seq
        self.pub_cmd_vel.publish(vel_cmd)

        # 等待新的激光数据（支持多层 fallback）
        try:
            data = self._wait_for_scan(timeout=0.5, require_new=True, last_seq=prev_scan_seq)
        except rospy.ROSException:
            try:
                data = self._wait_for_scan(timeout=0.0)
            except rospy.ROSException:
                data = self._try_scan_fallback(timeout_per_topic=0.5)

        state, done = self.getState(data, past_action)     
        executed_action = [linear_vel, ang_vel]
        reward, done = self.setReward(state, executed_action, past_action, done)

        # 发布可视化数据
        self._publish_goal()
        if latest_downsampled is not None:
            self._publish_risk_markers(latest_downsampled)

        # episode 结束时停止
        if done:
            self.pub_cmd_vel.publish(Twist())
        
        return np.asarray(state), reward, done

    
    def reset(self):
        """
        重置环境到初始状态
        
        流程：
        1. 调用 Gazebo reset_world 服务
        2. 等待激光数据
        3. 采样或刷新目标点位置
        4. 计算初始距离
        5. 重置计数器
        6. 构建初始状态
        
        Returns:
            np.ndarray: 初始状态向量
        """
        self.reset_count += 1
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = self._wait_for_scan(timeout=self.scan_wait_timeout)
        
        # 目标点采样策略
        if self.initGoal:
            # 首次初始化：安全采样，避免与障碍重叠
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.initGoal = False
        elif self.random_goal_on_reset and (self.reset_count % self.goal_reset_interval == 0):
            # 定期随机切换目标，避免陷入确定性局部最优
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
        else:
            # 保持目标不变，仅刷新可视化
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(False, delete=False)
           

        print("reset successfully")
        self.goal_distance = self.getGoalDistace()
        self.spin_streak = 0
        self.no_progress_streak = 0
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0

        # 重置轨迹
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        # 发布初始目标点
        self._publish_goal()

        state, _ = self.getState(data, [0]*self.action_dim)
        return np.asarray(state)
