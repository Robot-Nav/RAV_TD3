"""
目标点重新生成模块
===================
本模块负责管理仿真环境中目标点（goal_box）的生命周期，包含：
- 安全目标点采样（避免与障碍物重叠）
- 三层候选策略：严格安全 → 放宽安全 → 最优备选
- 目标模型在 Gazebo 中的生成、移动、删除、验证
- 视觉刷新策略：避免频繁删建造成的场景不同步
"""

import rospy
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Pose, Twist


class Respawn():
    """
    目标点重新生成管理器
    
    提供安全的随机目标点采样和 Gazebo 模型管理功能。
    """
    
    def __init__(self):
        """初始化目标点管理器"""
        # ========== 加载目标模型文件 ==========
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('scripts/TD3',
                                                'models/turtlebot3_square/goal_box/model.sdf')
        with open(self.modelPath, 'r') as f:
            self.model = f.read()
        self.goal_position = Pose()
        # 初始值仅作占位，首次 reset 会重新采样安全目标。
        self.init_goal_x = float(rospy.get_param('~init_goal_x', -4.0))
        self.init_goal_y = float(rospy.get_param('~init_goal_y', -0.2))

        # 预设候选目标点集合（结合地图障碍分布设计）
        self.goal_candidates = [
            (-5.4, -2.1), (-5.4, 0.0), (-5.4, 2.1), (-5.4, 4.3), (-5.4, 6.4),
            (-4.0, -2.1), (-4.0, -0.1), (-4.0, 2.2), (-4.0, 4.8), (-4.0, 6.8),
            (-2.6, -2.0), (-2.6, 0.0), (-2.6, 2.5), (-2.6, 4.6), (-2.6, 6.8),
            (-1.0, -2.0), (-1.0, 0.8), (-1.0, 3.1), (-1.0, 6.2),
            (0.8, -1.8), (0.8, 1.2), (0.8, 3.6), (0.8, 6.8),
            (5.6, -1.3), (5.6, 1.2), (5.6, 3.8), (5.6, 6.8),
        ]

        # ========== 采样边界与安全参数 ==========
        self.goal_min_x = rospy.get_param('~goal_min_x', -6.2)       # X 轴最小边界
        self.goal_max_x = rospy.get_param('~goal_max_x', 6.2)        # X 轴最大边界
        self.goal_min_y = rospy.get_param('~goal_min_y', -2.9)       # Y 轴最小边界
        self.goal_max_y = rospy.get_param('~goal_max_y', 7.8)        # Y 轴最大边界
        self.goal_boundary_margin = float(rospy.get_param('~goal_boundary_margin', 0.40))  # 边界安全余量
        self.goal_clearance = rospy.get_param('~goal_clearance', 0.60)                     # 障碍物安全净空
        self.goal_min_clearance = float(rospy.get_param('~goal_min_clearance', 0.25))      # 最小允许净空
        self.spawn_clearance = rospy.get_param('~spawn_clearance', 1.0)                    # 出生点安全距离
        self.max_goal_sample_trials = int(rospy.get_param('~max_goal_sample_trials', 400)) # 最大采样尝试次数
        self.goal_update_retries = max(1, int(rospy.get_param('~goal_update_retries', 4))) # 模型操作重试次数
        self.goal_verify_tolerance = float(rospy.get_param('~goal_verify_tolerance', 0.08)) # 位姿验证容差
        self.model_states_wait_timeout = float(rospy.get_param('~model_states_wait_timeout', 2.5)) # model_states 超时
        self.min_goal_update_interval = float(rospy.get_param('~min_goal_update_interval', 0.20))  # 最小更新间隔
        self.allow_random_goal_fallback = bool(rospy.get_param('~allow_random_goal_fallback', False)) # 允许随机回退
        self.random_candidate_samples = max(0, int(rospy.get_param('~random_candidate_samples', 120))) # 随机采样数量
        self.min_goal_switch_distance = float(rospy.get_param('~min_goal_switch_distance', 0.9))      # 最小切换距离

        # 初始化目标位置
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.goal_position.position.z = float(rospy.get_param('~goal_z', 0.04))
        self.goal_position.orientation.w = 1.0
        self.modelName = 'goal'
        
        # 障碍物检测参考点
        self.obstacle_1 = 0.3, 0.3
        self.obstacle_2 = 0.3, -0.3
        self.obstacle_3 = -0.3, 0.3
        self.obstacle_4 = -0.3, -0.3
        
        # 状态跟踪变量
        self.last_goal_x = self.init_goal_x    # 上一次目标 x 坐标
        self.last_goal_y = self.init_goal_y    # 上一次目标 y 坐标
        self.last_index = -1                   # 上一次选择的候选索引
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)  # 模型状态订阅
        self.model_positions = {}              # 当前场景中所有模型的位置字典
        self.check_model = False               # 目标模型是否存在标志
        self.index = 0                         # 候选目标索引
        self.index_num = 0                     # 候选总数
        self.R_num = 1                         # 轮次计数
        self._last_visual_update_time = 0.0    # 最后一次视觉更新时间

    def _wait_model_states(self, timeout=None, force=False):
        """
        等待 Gazebo model_states 消息
        
        Args:
            timeout: 等待超时时间
            force: True 则忽略缓存强制重新获取
            
        Returns:
            bool: 是否成功获取模型状态
        """
        if self.model_positions and not force:
            return True
        wait_timeout = self.model_states_wait_timeout if timeout is None else float(timeout)
        try:
            msg = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=wait_timeout)
            self.checkModel(msg)
            return bool(self.model_positions)
        except rospy.ROSException:
            rospy.logwarn_throttle(2.0, 'Timed out waiting for gazebo/model_states when sampling goal')
            return False

    def _goal_exists(self):
        """
        检查目标模型是否存在于场景中
        
        Returns:
            bool: True 表示目标模型存在
        """
        self._wait_model_states(timeout=0.4, force=True)
        return self.modelName in self.model_positions

    def _query_goal_pose(self, refresh=True):
        """
        查询目标模型的当前位姿
        
        Args:
            refresh: 是否刷新模型状态
            
        Returns:
            tuple: (x, y) 坐标，如果模型不存在则返回 None
        """
        if refresh:
            self._wait_model_states(timeout=0.4, force=True)
        if self.modelName not in self.model_positions:
            return None
        return self.model_positions[self.modelName]

    def _verify_goal_pose(self):
        """
        验证目标模型位姿是否与期望位置一致
        
        Returns:
            bool: True 表示位姿一致
        """
        for _ in range(self.goal_update_retries):
            pose = self._query_goal_pose(refresh=True)
            if pose is not None:
                cur_x, cur_y = pose
                dx = abs(cur_x - self.goal_position.position.x)
                dy = abs(cur_y - self.goal_position.position.y)
                if dx <= self.goal_verify_tolerance and dy <= self.goal_verify_tolerance:
                    return True
            time.sleep(0.03)
        return False

    def _wait_goal_pose(self, timeout=0.8):
        """
        等待目标模型位姿与期望位置对齐
        
        Args:
            timeout: 等待超时时间
            
        Returns:
            bool: True 表示位姿对齐成功
        """
        start = time.time()
        while (time.time() - start) < timeout and not rospy.is_shutdown():
            pose = self._query_goal_pose(refresh=True)
            if pose is not None:
                cur_x, cur_y = pose
                dx = abs(cur_x - self.goal_position.position.x)
                dy = abs(cur_y - self.goal_position.position.y)
                if dx <= self.goal_verify_tolerance and dy <= self.goal_verify_tolerance:
                    return True
            time.sleep(0.03)
        return False

    def _force_recreate_model(self):
        """
        兜底方案：删除后重建目标模型
        
        仅在 set_model_state 失败时使用，避免高频删建触发 GUI 竞态。
        
        Returns:
            bool: True 表示重建成功
        """
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)

        for _ in range(self.goal_update_retries):
            self._wait_model_states(timeout=0.4, force=True)
            if self.modelName in self.model_positions:
                try:
                    del_model_prox(self.modelName)
                except Exception:
                    pass
                time.sleep(0.08)

            try:
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, 'world')
            except Exception as e:
                # 处理并发状态下 "already exists" 的残留
                if 'already exists' in str(e):
                    try:
                        del_model_prox(self.modelName)
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            if self._wait_goal_pose(timeout=1.0):
                return True

        # 最后再尝试一次"只生成不删除"，避免偶发时序把模型停留在不存在状态
        try:
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, 'world')
        except Exception:
            pass
        return self._wait_goal_pose(timeout=1.0)

    def _spawn_goal_model(self):
        """
        在目标位置生成目标模型
        
        Returns:
            bool: True 表示生成成功
        """
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)

        for _ in range(self.goal_update_retries):
            try:
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, 'world')
                if self._wait_goal_pose(timeout=1.0):
                    return True
            except Exception as e:
                # 并发情况下可能已存在，直接走 set_pose 路径
                if 'already exists' in str(e):
                    if self._set_goal_pose():
                        return True
                time.sleep(0.05)

        return self._wait_goal_pose(timeout=0.6)

    def _set_goal_pose(self):
        """
        通过 set_model_state 服务移动目标模型到新位置
        
        Returns:
            bool: True 表示移动成功
        """
        gx = float(self.goal_position.position.x)
        gy = float(self.goal_position.position.y)
        gz = float(self.goal_position.position.z)
        # 位姿有效性校验
        if (not math.isfinite(gx)) or (not math.isfinite(gy)) or (not math.isfinite(gz)):
            rospy.logwarn('Skip goal pose update due to non-finite pose: (%.3f, %.3f, %.3f)', gx, gy, gz)
            return False

        rospy.wait_for_service('gazebo/set_model_state')
        set_model_state_prox = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        state = ModelState()
        state.model_name = self.modelName
        state.pose = self.goal_position
        state.twist = Twist()
        state.reference_frame = 'world'

        for _ in range(self.goal_update_retries):
            try:
                resp = set_model_state_prox(state)
                if getattr(resp, 'success', False):
                    # 等待 model_states 刷新以确保视觉更新
                    time.sleep(0.05)
                    if self._wait_goal_pose(timeout=0.6):
                        return True
            except Exception:
                pass
            time.sleep(0.04)

        return False

    def checkModel(self, model):
        """
        模型状态回调函数，解析场景中所有模型的位置
        
        Args:
            model: ModelStates 消息
        """
        self.check_model = False
        self.model_positions = {}
        for i in range(len(model.name)):
            name = model.name[i]
            self.model_positions[name] = (
                model.pose[i].position.x,
                model.pose[i].position.y,
            )
            if name == "goal":
                self.check_model = True

    def _obstacle_radius(self, name):
        """
        根据模型名称估算障碍物的等效半径
        
        Args:
            name: 模型名称
            
        Returns:
            float: 等效半径（米）
        """
        if name.startswith('moving_obstacle'):
            return 0.95
        if name.startswith('box'):
            return 1.05
        if name.startswith('cylinder'):
            return 0.45
        if name.startswith('barrier'):
            return 0.55
        if name.startswith('pillar'):
            return 0.30
        if name == 'limo':
            return 0.80
        return 0.55

    def _goal_safety_margin(self, x, y, clearance):
        """
        计算候选目标点的安全净空值
        
        安全净空 = min(距离边界, 距离出生点, 距离所有障碍物) - 所需最小净空
        
        Args:
            x: 候选目标 x 坐标
            y: 候选目标 y 坐标
            clearance: 所需最小净空
            
        Returns:
            float: 安全净空值（>0 表示安全，<0 表示不安全）
        """
        if not self.model_positions:
            return -1e6

        x_min = self.goal_min_x + self.goal_boundary_margin
        x_max = self.goal_max_x - self.goal_boundary_margin
        y_min = self.goal_min_y + self.goal_boundary_margin
        y_max = self.goal_max_y - self.goal_boundary_margin

        # 边界检查
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return -1e6

        # 与边界、出生点的最小净空值
        min_margin = min(x - x_min, x_max - x, y - y_min, y_max - y)
        min_margin = min(min_margin, math.hypot(x, y) - self.spawn_clearance)

        # 与所有障碍物的最小净空值
        for name, (ox, oy) in self.model_positions.items():
            if name in ('goal', 'ground_plane', 'turtlebot3_square_0', 'sun'):
                continue
            limit = self._obstacle_radius(name) + clearance
            margin = math.hypot(x - ox, y - oy) - limit
            if margin < min_margin:
                min_margin = margin

        return min_margin

    def _is_goal_safe(self, x, y, clearance=None):
        """
        检查候选目标点是否安全
        
        Args:
            x: 候选目标 x 坐标
            y: 候选目标 y 坐标
            clearance: 所需最小净空
            
        Returns:
            bool: True 表示安全
        """
        if clearance is None:
            clearance = self.goal_clearance
        return self._goal_safety_margin(x, y, clearance) >= 0.0

    def _sample_random_safe_goal(self, clearance=None):
        """
        随机采样安全目标点
        
        Args:
            clearance: 所需最小净空
            
        Returns:
            tuple: (x, y) 坐标，如果采样失败则返回 None
        """
        if clearance is None:
            clearance = self.goal_clearance
        for _ in range(self.max_goal_sample_trials):
            gx = random.uniform(self.goal_min_x, self.goal_max_x)
            gy = random.uniform(self.goal_min_y, self.goal_max_y)
            if self._is_goal_safe(gx, gy, clearance=clearance):
                return gx, gy
        return None

    def _pick_diverse_candidate(self, candidates):
        """
        从候选集合中选择多样化目标点
        
        策略：
        1. 过滤掉距离上次目标太近的点（鼓励探索）
        2. 在剩余候选中按净空值做加权随机（避免固定在同一点）
        
        Args:
            candidates: [(margin, key, gx, gy), ...]
            
        Returns:
            tuple: 选中的候选项
        """
        if not candidates:
            return None

        # 过滤：保留距离上次目标足够远的候选
        diverse = []
        for item in candidates:
            _, _, gx, gy = item
            if math.hypot(gx - self.last_goal_x, gy - self.last_goal_y) >= self.min_goal_switch_distance:
                diverse.append(item)

        pool = diverse if diverse else candidates
        # 按净空值加权随机
        weights = [max(0.01, item[0] + 0.05) for item in pool]
        return random.choices(pool, weights=weights, k=1)[0]
   
   
    def respawnModel(self):
        """
        重新生成目标模型
        
        Returns:
            bool: True 表示操作成功
        """
        if self._goal_exists():
            if self._set_goal_pose():
                return True
            return False
        return self._spawn_goal_model()

    def deleteModel(self):
        """从场景中删除目标模型"""
        if not self._goal_exists():
            return
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        try:
            del_model_prox(self.modelName)
        except Exception:
            pass

    def moveModel(self):
        """
        移动目标模型到新位置
        
        Returns:
            bool: True 表示操作成功
        """
        if self._set_goal_pose():
            return True
        if not self._goal_exists():
            return self._spawn_goal_model()
        return False

    def _update_goal_visual(self, force=False):
        """
        更新目标模型的视觉显示
        
        Args:
            force: True 则强制刷新（跳过最小更新间隔检查）
            
        Returns:
            bool: True 表示更新成功
        """
        now = time.time()
        # 只有非强制模式下才检查最小更新间隔
        if not force and self._goal_exists() and (now - self._last_visual_update_time) < self.min_goal_update_interval:
            return True

        # 优先使用 set_model_state 移动，避免频繁删建导致的场景错误
        for _ in range(self.goal_update_retries):
            if self._goal_exists():
                # 模型存在，直接移动
                ok = self._set_goal_pose()
            else:
                # 模型不存在，需要创建
                ok = self._spawn_goal_model()

            if ok and self._wait_goal_pose(timeout=0.6):
                self._last_visual_update_time = time.time()
                rospy.loginfo(
                    'Goal position : %.1f, %.1f',
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                )
                return True
            time.sleep(0.05)

        rospy.logwarn(
            'Goal visual update failed after retries, target pose=(%.2f, %.2f)',
            self.goal_position.position.x,
            self.goal_position.position.y,
        )
        return False

    def getPosition(self, position_check=False, delete=False):
        """
        获取新的目标点位置
        
        采样策略（三层回退）：
        1. 严格安全检查（clearance 完整值）
        2. 放宽安全检查（clearance × 0.65）
        3. 最优备选（选择净空最大的点）
        
        Args:
            position_check: True 则执行安全检查
            delete: True 则强制刷新视觉
            
        Returns:
            tuple: (goal_x, goal_y)
        """
        requested_position_check = bool(position_check)
        # delete 请求仅作为"强制刷新"信号，不再执行删除
        force_refresh = bool(delete)

        if position_check:
            if not self._wait_model_states(force=True):
                rospy.logwarn_throttle(
                    2.0,
                    'model_states unavailable during goal sampling, keep previous goal (%.2f, %.2f)',
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                )
                position_check = False

        while position_check:
            # 三层策略：严格安全 → 放宽安全（仍不重叠） → 最优备选
            relaxed_clearance = max(self.goal_min_clearance, self.goal_clearance * 0.65)
            candidate_pool = []
            # 添加预设候选点
            for idx, (gx, gy) in enumerate(self.goal_candidates):
                candidate_pool.append(('fixed:%d' % idx, gx, gy))

            # 添加随机采样点
            for ridx in range(self.random_candidate_samples):
                gx = random.uniform(self.goal_min_x, self.goal_max_x)
                gy = random.uniform(self.goal_min_y, self.goal_max_y)
                candidate_pool.append(('rand:%d' % ridx, gx, gy))

            strict_safe = []    # 严格安全检查通过的候选
            relaxed_safe = []   # 放宽安全检查通过的候选
            best_effort = []    # 所有候选（用于兜底）

            for key, gx, gy in candidate_pool:
                strict_margin = self._goal_safety_margin(gx, gy, self.goal_clearance)
                relaxed_margin = self._goal_safety_margin(gx, gy, relaxed_clearance)

                # 最优备选用于极端情况下兜底
                best_effort.append((relaxed_margin, key, gx, gy))

                if strict_margin >= 0.0:
                    strict_safe.append((strict_margin, key, gx, gy))
                elif relaxed_margin >= 0.0:
                    relaxed_safe.append((relaxed_margin, key, gx, gy))

            selected = False

            # 第一层：严格安全检查
            if strict_safe:
                pick = self._pick_diverse_candidate(strict_safe)
                if pick is not None:
                    _, key, gx, gy = pick
                    self.last_index = key
                    self.goal_position.position.x = gx
                    self.goal_position.position.y = gy
                    selected = True

            # 第二层：放宽安全检查
            if (not selected) and relaxed_safe:
                pick = self._pick_diverse_candidate(relaxed_safe)
                if pick is not None:
                    _, key, gx, gy = pick
                    self.last_index = key
                    self.goal_position.position.x = gx
                    self.goal_position.position.y = gy
                    rospy.logwarn_throttle(2.0, 'Using relaxed goal clearance %.2f for goal sampling', relaxed_clearance)
                    selected = True

            # 第三层：随机采样回退
            if (not selected) and self.allow_random_goal_fallback:
                sampled = self._sample_random_safe_goal(clearance=relaxed_clearance)
                if sampled is not None:
                    self.goal_position.position.x, self.goal_position.position.y = sampled
                    selected = True

            # 第四层：最优备选（选择净空最大的点）
            if (not selected) and best_effort:
                best_effort.sort(key=lambda item: item[0], reverse=True)
                pick = self._pick_diverse_candidate(best_effort[:max(1, min(20, len(best_effort)))])
                if pick is None:
                    pick = best_effort[0]

                margin, key, gx, gy = pick
                self.last_index = key
                self.goal_position.position.x = gx
                self.goal_position.position.y = gy
                rospy.logwarn_throttle(
                    2.0,
                    'Fallback to best-effort goal (margin %.3f), check map clearances',
                    margin,
                )
                selected = True

            if selected:
                strict_margin_dbg = self._goal_safety_margin(
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                    self.goal_clearance,
                )
                relaxed_clearance_dbg = max(self.goal_min_clearance, self.goal_clearance * 0.65)
                relaxed_margin_dbg = self._goal_safety_margin(
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                    relaxed_clearance_dbg,
                )
                rospy.loginfo(
                    'Selected goal=(%.2f, %.2f), margin(strict=%.3f, relaxed=%.3f), pool_size=%d',
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                    strict_margin_dbg,
                    relaxed_margin_dbg,
                    len(candidate_pool),
                )
                position_check = False
            else:
                rospy.logwarn_throttle(
                    2.0,
                    'No candidate goal available, keeping previous goal (%.2f, %.2f)',
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                )
                position_check = False

        time.sleep(0.1)

        # 检查目标位置是否发生变化
        goal_moved = (abs(self.goal_position.position.x - self.last_goal_x) > 0.01 or
                      abs(self.goal_position.position.y - self.last_goal_y) > 0.01)

        # 仅在"目标改变 / 主动删除 / 模型不存在 / 位姿不一致"时刷新可视化
        # 避免每次 reset 都删建导致偶发看不见
        need_refresh = False
        force_refresh_visual = False
        if force_refresh or requested_position_check:
            need_refresh = True
            force_refresh_visual = True
        elif goal_moved:
            # 目标位置发生变化，需要强制刷新
            need_refresh = True
            force_refresh_visual = True
        else:
            if not self._goal_exists():
                need_refresh = True
            elif not self._wait_goal_pose(timeout=0.25):
                need_refresh = True

        if need_refresh:
            self._update_goal_visual(force=force_refresh_visual)
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
