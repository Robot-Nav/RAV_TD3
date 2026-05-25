"""
================================================================================
优先经验回放缓冲区模块（Prioritized Experience Replay, PER）
================================================================================
功能说明：
    实现基于 TD-error 的优先级经验回放机制，核心特性包括：
    1. SumTree 数据结构：支持 O(log N) 的优先级更新和采样
    2. 重要性采样权重（IS Weights）：纠正优先级采样引入的分布偏差
    3. 优先级归一化：防止数值溢出，确保训练稳定性

设计原理：
    - 传统经验回放采用均匀随机采样，忽略了不同经验的学习价值差异
    - PER 根据 TD-error 大小分配优先级，让智能体更多学习"意外"的经验
    - 通过重要性采样权重补偿，保证算法收敛性
"""

import numpy as np


class SumTree:
    """
    SumTree（求和树）数据结构
    
    核心功能：
        高效维护带权重的数据集合，支持 O(log N) 的优先级更新和采样
    
    结构说明：
        - 完全二叉树结构，使用数组存储
        - 叶子节点（索引 capacity-1 到 2*capacity-2）：存储实际数据的优先级
        - 内部节点（索引 0 到 capacity-2）：存储子树优先级之和
        - 根节点（索引 0）：存储所有叶子节点的优先级总和
    
    时间复杂度：
        - 添加元素：O(log N)
        - 更新优先级：O(log N)
        - 按优先级采样：O(log N)
    
    Attributes:
        capacity (int): 叶子节点容量，即最多存储的数据条数
        tree (np.ndarray): 树结构数组，存储各节点的优先级和值
        data (np.ndarray): 数据存储数组，存储实际的经验索引
        n_entries (int): 当前已存储的数据条目数
        write (int): 当前写入位置指针（环形缓冲区）
    """
    
    def __init__(self, capacity):
        """
        初始化 SumTree
        
        Args:
            capacity (int): 缓冲区最大容量，决定叶子节点数量
            
        Note:
            树的总节点数为 2*capacity-1（capacity个叶子 + capacity-1个内部节点）
            使用 float64 类型存储优先级，防止大数累加时的精度损失
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, delta):
        """
        向上传播优先级变化量
        
        功能说明：
            当叶子节点的优先级发生变化时，递归更新所有父节点的和值
        
        算法流程：
            1. 计算当前节点的父节点索引：(idx - 1) // 2
            2. 将变化量 delta 加到父节点的值上
            3. 如果父节点不是根节点，继续向上传播
        
        Args:
            idx (int): 发生变化的节点索引（叶子节点或内部节点）
            delta (float): 优先级的变化量（新值 - 旧值）
            
        Time Complexity: O(log N)
        """
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent > 0:
            self._propagate(parent, delta)

    def add(self, priority, data):
        """
        添加新数据到 SumTree
        
        功能说明：
            将新数据及其优先级添加到树中，采用环形缓冲区策略
        
        算法流程：
            1. 计算叶子节点索引：write + capacity - 1
            2. 存储数据到 data 数组
            3. 调用 update 方法设置优先级（会触发向上传播）
            4. 更新 write 指针，超出 capacity 时回到 0（环形）
            5. 更新 n_entries 计数
        
        Args:
            priority (float): 新数据的初始优先级，必须为正数
            data (object): 要存储的数据，通常是经验在缓冲区中的索引
            
        Time Complexity: O(log N)
        
        Note:
            当缓冲区满时，新数据会覆盖最老的数据（环形覆盖策略）
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """
        更新指定节点的优先级
        
        功能说明：
            修改树中任意节点的优先级，并向上传播变化量
        
        算法流程：
            1. 计算优先级变化量：delta = 新优先级 - 原优先级
            2. 更新节点的优先级值
            3. 调用 _propagate 向上传播变化量
        
        Args:
            idx (int): 要更新的节点索引
            priority (float): 新的优先级值，必须为正数
            
        Time Complexity: O(log N)
        
        Note:
            如果 priority 为 0，该节点将永远不会被采样到
        """
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s):
        """
        按优先级累积和采样
        
        功能说明：
            根据给定的累积值 s，从根节点开始向下遍历，找到对应的叶子节点
            这是 PER 按优先级采样的核心操作
        
        算法流程：
            1. 从根节点（索引 0）开始
            2. 比较 s 与左子树的值：
               - 如果 s <= 左子树值，进入左子树
               - 否则 s -= 左子树值，进入右子树
            3. 重复直到到达叶子节点
        
        Args:
            s (float): 采样值，范围应在 [0, total_priority) 之间
            
        Returns:
            tuple: (leaf_idx, priority)
                - leaf_idx (int): 采样到的叶子节点索引
                - priority (float): 该叶子节点的优先级值
                
        Time Complexity: O(log N)
        
        Example:
            假设总优先级为 100，要采样 3 个经验：
            - 生成随机数 s1=23, s2=56, s3=89
            - 分别调用 get(s1), get(s2), get(s3) 得到对应的经验
        """
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx, self.tree[idx]

    def total(self):
        """
        获取总优先级
        
        Returns:
            float: 所有叶子节点优先级的总和（根节点的值）
            
        Note:
            总优先级用于计算采样概率：P(i) = priority_i / total_priority
        """
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    核心功能：
        基于 TD-error 的优先级经验采样，让智能体更多学习"意外"的经验
    
    设计原理：
        1. 新经验赋予最大优先级，确保至少被采样一次
        2. 训练后根据新的 TD-error 更新优先级
        3. 使用重要性采样权重纠正分布偏差
    
    关键参数：
        alpha (float): 优先级指数，控制优先级的激进程度
            - 0.0: 均匀采样（等同于普通经验回放）
            - 1.0: 完全按优先级采样
            - 0.6: 推荐值，平衡探索与利用
        epsilon (float): 小常数，防止优先级为 0
    
    Attributes:
        mem_size (int): 缓冲区最大容量
        batch_size (int): 每次采样的批次大小
        tree (SumTree): SumTree 数据结构，维护经验优先级
        alpha (float): 优先级指数
        epsilon (float): 防止除零的小常数
        state_memory (np.ndarray): 状态存储数组
        action_memory (np.ndarray): 动作存储数组
        reward_memory (np.ndarray): 奖励存储数组
        next_state_memory (np.ndarray): 下一状态存储数组
        terminal_memory (np.ndarray): 终止标志存储数组
        mem_cnt (int): 当前已存储的经验数量
        max_p (float): 当前最大优先级，用于新经验的初始化
    """
    
    def __init__(self, max_size, state_dim, action_dim, batch_size, alpha=0.6, epsilon=0.01,
                 beta_start=0.4, beta_increment_per_sampling=1e-4):
        """
        初始化优先经验回放缓冲区
        
        Args:
            max_size (int): 缓冲区最大容量
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            batch_size (int): 每次采样的批次大小
            alpha (float, optional): 优先级指数，默认 0.6
            epsilon (float, optional): 防止除零的小常数，默认 0.01
            
        Note:
            alpha 越大，采样越倾向于高 TD-error 的经验
            epsilon 确保即使 TD-error 为 0，优先级也不为 0
        """
        self.mem_size = max_size
        self.batch_size = batch_size
        self.tree = SumTree(max_size)
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros((max_size,), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminal_memory = np.zeros((max_size,), dtype=bool)
        
        self.mem_cnt = 0
        self.max_p = 1.0
        self.beta = float(beta_start)
        self.beta_increment_per_sampling = float(beta_increment_per_sampling)

    def store_transition(self, state, action, reward, state_, done):
        """
        存储经验到缓冲区
        
        功能说明：
            将新的经验 (s, a, r, s', done) 存储到缓冲区，并赋予最大优先级
        
        算法流程：
            1. 计算存储位置：mem_cnt % mem_size（环形缓冲区）
            2. 存储状态、动作、奖励、下一状态、终止标志
            3. 使用当前最大优先级 max_p 作为新经验的优先级
            4. 更新 mem_cnt 计数
        
        Args:
            state (np.ndarray): 当前状态
            action (np.ndarray): 执行的动作
            reward (float): 获得的奖励
            state_ (np.ndarray): 下一状态
            done (bool): 是否终止
            
        Note:
            新经验使用最大优先级，确保至少被采样一次
            如果缓冲区已满，会覆盖最老的经验
        """
        mem_idx = self.mem_cnt % self.mem_size
        
        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
        
        priority = self.max_p
        self.tree.add(priority, mem_idx)
        
        self.mem_cnt += 1

    def sample_buffer(self):
        """
        按优先级采样经验批次
        
        功能说明：
            使用 SumTree 按优先级采样 batch_size 条经验，并计算重要性采样权重
        
        算法流程：
            1. 获取总优先级，检查异常情况（<=0, inf, nan）
            2. 将总优先级分成 batch_size 个区间
            3. 每个区间内均匀随机采样一个值
            4. 使用 SumTree.get() 找到对应的经验
            5. 计算采样概率和重要性采样权重
        
        Returns:
            tuple: (states, actions, rewards, states_, terminals, indices, weights)
                - states (np.ndarray): 采样状态批次
                - actions (np.ndarray): 采样动作批次
                - rewards (np.ndarray): 采样奖励批次
                - states_ (np.ndarray): 采样下一状态批次
                - terminals (np.ndarray): 采样终止标志批次
                - indices (np.ndarray): 采样经验在树中的索引（用于后续更新优先级）
                - weights (np.ndarray): 重要性采样权重（用于纠正分布偏差）
                
        Time Complexity: O(batch_size * log N)
        
        Note:
            重要性采样权重 = (N * P(i))^(-alpha) / max_weight
            用于在梯度更新时降低高优先级样本的影响，保证收敛性
        """
        n_batch = self.batch_size
        batch = np.zeros(self.batch_size, dtype=np.int64)
        indices = np.zeros(self.batch_size, dtype=np.int64)
        priorities = np.zeros(self.batch_size, dtype=np.float64)
        
        total_priority = self.tree.total()
        current_size = min(self.mem_cnt, self.mem_size)

        # 异常情况回退：当优先级总和不可用时，退化到均匀采样，避免SumTree状态被破坏
        if total_priority <= 0 or np.isinf(total_priority) or np.isnan(total_priority):
            batch = np.random.choice(current_size, self.batch_size, replace=False)
            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            states_ = self.next_state_memory[batch]
            terminals = self.terminal_memory[batch]
            indices = batch.astype(np.int64)
            weights = np.ones(self.batch_size, dtype=np.float32)
            return states, actions, rewards, states_, terminals, indices, weights
        
        # 将总优先级分成 batch_size 个区间，每个区间内均匀采样
        segment = total_priority / n_batch
        for i in range(n_batch):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            # 防止采样值越界
            if s >= total_priority:
                s = total_priority - 1e-6
            
            idx, priority = self.tree.get(s)
            data_slot = idx - self.tree.capacity + 1
            data_slot = max(0, min(data_slot, self.mem_size - 1))
            indices[i] = data_slot
            priorities[i] = max(priority, 1e-6)

            sampled_mem_idx = self.tree.data[data_slot]
            if sampled_mem_idx is None:
                sampled_mem_idx = 0
            batch[i] = int(sampled_mem_idx)

        # 计算采样概率和重要性采样权重
        probs = priorities / total_priority
        probs = np.clip(probs, 1e-6, None)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights = (current_size * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-6)
        
        # 从存储数组中提取采样数据
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        根据新的 TD-error 更新经验优先级
        
        功能说明：
            训练后，根据计算出的 TD-error 更新采样到的经验的优先级
        
        算法流程：
            1. 计算新优先级：(|TD-error| + epsilon)^alpha
            2. 裁剪优先级范围：[epsilon, 10.0]，防止数值溢出
            3. 更新 SumTree 中对应节点的优先级
            4. 更新 max_p 为当前最大优先级
        
        Args:
            indices (np.ndarray): 要更新的经验在树中的索引数组
            td_errors (np.ndarray): 对应的 TD-error 数组
            
        Note:
            TD-error 越大，优先级越高，该经验被采样的概率越大
            优先级上限设为 10.0，防止极端 TD-error 导致数值溢出
        """
        priorities = np.clip(np.abs(td_errors), self.epsilon, None) ** self.alpha
        priorities = np.clip(priorities, self.epsilon, 10.0)
        
        for idx, p in zip(indices, priorities):
            tree_idx = idx + self.tree.capacity - 1
            if tree_idx >= 0 and tree_idx < len(self.tree.tree):
                self.tree.update(tree_idx, p)
        
        if len(priorities) > 0:
            self.max_p = max(self.max_p, priorities.max())

    def ready(self):
        """
        检查缓冲区是否准备好进行采样
        
        Returns:
            bool: 如果当前存储的经验数量 >= batch_size，返回 True
            
        Note:
            在训练开始前，需要确保缓冲区中有足够的经验
        """
        return self.mem_cnt >= self.batch_size
