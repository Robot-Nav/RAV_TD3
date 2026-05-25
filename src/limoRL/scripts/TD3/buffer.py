#!/usr/bin/python3
"""
经验回放缓冲区模块
==================
实现 TD3 算法所需的环形经验回放缓冲区（Replay Buffer），支持：
- 固定容量的环形存储（FIFO 淘汰策略）
- 随机采样（无放回抽样）
- 经验就绪检测
"""

import numpy as np


class ReplayBuffer:
    """
    环形经验回放缓冲区
    
    用于存储智能体与环境交互的经验元组 (s, a, r, s', done)，
    训练时从中随机采样小批量数据以打破数据相关性。
    """
    
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        """
        初始化缓冲区
        
        Args:
            max_size: 缓冲区最大容量
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            batch_size: 训练时的采样批量大小
        """
        self.mem_size = max_size       # 最大存储容量
        self.batch_size = batch_size   # 采样批量大小
        self.mem_cnt = 0               # 累计存储的经验数量（用于环形索引计算）
        
        # 预分配 NumPy 数组，提升存储/读取性能
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)     # 状态 s
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)   # 动作 a
        self.reward_memory = np.zeros((max_size,), dtype=np.float32)              # 奖励 r
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32) # 下一状态 s'
        self.terminal_memory = np.zeros((max_size,), dtype=bool)                  # 终止标志 done

    def store_transition(self, state, action, reward, state_, done):
        """
        存储一条经验元组
        
        使用环形缓冲区策略：当缓冲区满时，覆盖最旧的经验。
        
        Args:
            state: 当前状态 (ndarray)
            action: 执行的动作 (ndarray)
            reward: 获得的奖励 (float)
            state_: 下一状态 (ndarray)
            done: 是否终止 (bool)
        """
        mem_idx = self.mem_cnt % self.mem_size  # 环形索引

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):
        """
        随机采样一个小批量经验数据
        
        使用无放回抽样（replace=False），确保批次内样本不重复。
        
        Returns:
            tuple: (states, actions, rewards, states_, terminals)
                每个元素都是形状为 (batch_size, dim) 的 NumPy 数组
        """
        mem_len = min(self.mem_cnt, self.mem_size)  # 实际可用样本数
        batch = np.random.choice(mem_len, self.batch_size, replace=False)  # 随机采样索引

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        """
        检查缓冲区是否已满 enough 进行采样
        
        Returns:
            bool: True 表示已存储至少 batch_size 条经验
        """
        return self.mem_cnt >= self.batch_size
