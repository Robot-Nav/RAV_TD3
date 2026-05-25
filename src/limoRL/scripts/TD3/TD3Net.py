"""
TD3 算法核心模块（Twin Delayed Deep Deterministic Policy Gradient）
====================================================================
本模块实现了改进版 TD3 算法，包含以下特性：
- 双 Actor 策略网络 + 投票集成决策
- 风险感知注意力编码器（Risk-Aware Attention Encoder）
- 双 Critic 价值网络（Clipped Double Q-Learning）
- 延迟策略更新（Delayed Policy Updates）
- 策略噪声平滑（Policy Smoothing）
- 共识正则化（Consensus Regularization）
"""

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
from .buffer import ReplayBuffer
from .per_buffer import PrioritizedReplayBuffer

# 自动选择计算设备：优先使用 CUDA，否则使用 CPU
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print("device: ",device)


def _safe_load_state_dict(checkpoint_file):
    """
    安全加载模型权重
    
    优先使用 weights_only 模式避免不安全的 pickle 执行警告，
    同时兼容旧版 PyTorch（不支持 weights_only 参数）。
    
    Args:
        checkpoint_file: 模型权重文件路径
        
    Returns:
        dict: 模型权重字典
    """
    try:
        return T.load(checkpoint_file, map_location=device, weights_only=True)
    except TypeError:
        return T.load(checkpoint_file, map_location=device)


class PolarLidarAttentionEncoder(nn.Module):
    """
    极坐标激光雷达注意力编码器
    
    将激光雷达扫描数据和目标极坐标信息融合为高维特征向量。
    
    核心设计：
    1. 扇区特征提取：将每个激光扇区映射为嵌入向量
    2. 风险感知注意力：结合学习注意力权重和显式距离风险偏置
    3. 目标特征编码：通过 MLP 处理目标朝向和距离
    4. 统计特征融合：融合扫描数据的最小值、均值、标准差
    
    输入: [激光扇区数据(N维) + 目标极坐标(2维)]
    输出: (融合特征向量, 注意力权重)
    """
    
    def __init__(self, scan_dim, goal_dim, sector_embed_dim=48, fusion_dim=192):
        """
        Args:
            scan_dim: 激光雷达扇区数量
            goal_dim: 目标极坐标维度（heading + distance）
            sector_embed_dim: 扇区嵌入向量维度
            fusion_dim: 融合特征维度
        """
        super(PolarLidarAttentionEncoder, self).__init__()
        self.scan_dim = scan_dim
        self.goal_dim = goal_dim

        # 扇区特征提取层：1维距离 → sector_embed_dim 维嵌入
        self.sector_fc = nn.Linear(1, sector_embed_dim)
        self.sector_ln = nn.LayerNorm(sector_embed_dim)
        
        # 注意力权重计算层
        self.attn_fc = nn.Linear(sector_embed_dim, 1)
        self.risk_bias_gain = nn.Parameter(T.tensor(1.5, dtype=T.float))  # 风险偏置增益系数

        # 目标极坐标编码器（heading + distance → 64维特征）
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 特征融合层：拼接上下文向量 + 目标特征 + 统计特征
        self.fusion = nn.Linear(sector_embed_dim + 64 + 3, fusion_dim)
        self.fusion_ln = nn.LayerNorm(fusion_dim)

    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, scan_dim + goal_dim]
            
        Returns:
            tuple: (fused_feature, attention_weights)
        """
        # 分离激光数据和目标数据
        scan = state[:, :self.scan_dim]
        goal = state[:, self.scan_dim:self.scan_dim + self.goal_dim]

        # --- 扇区特征提取 ---
        # [B, N, 1] → [B, N, E]
        scan_seq = scan.unsqueeze(-1)
        sector_embed = self.sector_fc(scan_seq)
        sector_embed = self.sector_ln(sector_embed)
        sector_embed = F.relu(sector_embed)

        # --- 风险感知注意力机制 ---
        # 1. 学习到的注意力权重
        learned_logits = self.attn_fc(sector_embed).squeeze(-1)
        # 2. 显式风险偏置：距离越近，权重越大（倒数关系）
        risk_bias = self.risk_bias_gain / (scan + 0.05)
        # 3. 融合两种注意力信号
        attn_logits = learned_logits + risk_bias
        attn_weights = F.softmax(attn_logits, dim=-1)

        # 4. 加权求和得到上下文向量
        context = T.bmm(attn_weights.unsqueeze(1), sector_embed).squeeze(1)

        # --- 全局统计特征提取 ---
        scan_min = T.min(scan, dim=1, keepdim=True).values    # 最小距离（最近障碍物）
        scan_mean = T.mean(scan, dim=1, keepdim=True)          # 平均距离（空间开阔度）
        scan_std = T.std(scan, dim=1, keepdim=True, unbiased=False)  # 距离标准差（障碍物分布均匀度）
        scan_stats = T.cat([scan_min, scan_mean, scan_std], dim=1)

        # --- 目标特征编码 ---
        goal_feat = self.goal_mlp(goal)
        
        # --- 特征融合 ---
        fused = T.cat([context, goal_feat, scan_stats], dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_ln(fused)
        fused = F.relu(fused)
        return fused, attn_weights


class ActorNetwork(nn.Module):
    """
    策略网络（Actor）
    
    将状态映射为连续动作 [线速度, 角速度]。
    使用风险感知注意力编码器提取特征，然后通过两层全连接网络输出动作。
    """
    
    def __init__(self, alpha, scan_dim, goal_dim, action_dim, fc1_dim, fc2_dim, action_limit_v, action_limit_w):
        """
        Args:
            alpha: 学习率
            scan_dim: 激光雷达扇区数量
            goal_dim: 目标极坐标维度
            action_dim: 动作空间维度（通常为2）
            fc1_dim: 第一层全连接网络维度（融合层输出）
            fc2_dim: 第二层全连接网络维度
            action_limit_v: 线速度上限
            action_limit_w: 角速度上限
        """
        super(ActorNetwork, self).__init__()
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        # 编码器：状态 → 高维特征
        self.encoder = PolarLidarAttentionEncoder(scan_dim=scan_dim, goal_dim=goal_dim, fusion_dim=fc1_dim)
        # 中间层
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        # 输出层
        self.fc3 = nn.Linear(fc2_dim, action_dim)

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            Tensor: 动作张量 [batch_size, action_dim]
        """
        x, _ = self.encoder(state)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        action_raw = self.fc3(x)

        # 输出激活函数：确保动作在合法范围内
        # 线速度：sigmoid → [0, action_limit_v]（只能前进）
        # 角速度：tanh → [-action_limit_w, action_limit_w]（可左右转向）
        action = T.zeros_like(action_raw)
        action[:, 0] = T.sigmoid(action_raw[:, 0]) * self.action_limit_v
        action[:, 1] = T.tanh(action_raw[:, 1]) * self.action_limit_w
        return action

    def save_checkpoint(self, checkpoint_file):
        """保存模型权重到文件"""
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """从文件加载模型权重"""
        self.load_state_dict(_safe_load_state_dict(checkpoint_file))


class CriticNetwork(nn.Module):
    """
    价值网络（Critic）
    
    评估状态-动作对的价值 Q(s, a)，返回标量 Q 值。
    使用双 Critic 架构，训练时取最小值以避免 Q 值高估。
    """
    
    def __init__(self, beta, scan_dim, goal_dim, action_dim, fc1_dim, fc2_dim):
        """
        Args:
            beta: 学习率
            scan_dim: 激光雷达扇区数量
            goal_dim: 目标极坐标维度
            action_dim: 动作空间维度
            fc1_dim: 编码器融合维度
            fc2_dim: 第二层全连接网络维度
        """
        super(CriticNetwork, self).__init__()
        # 共享的编码器
        self.encoder = PolarLidarAttentionEncoder(scan_dim=scan_dim, goal_dim=goal_dim, fusion_dim=fc1_dim)
        # 动作特征投影层
        self.action_fc = nn.Linear(action_dim, 64)
        # 状态-动作融合层
        self.fc2 = nn.Linear(fc1_dim + 64, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        # Q 值输出层（标量）
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            Tensor: Q 值张量 [batch_size, 1]
        """
        s_feat, _ = self.encoder(state)      # 状态特征
        a_feat = F.relu(self.action_fc(action))  # 动作特征
        x = T.cat([s_feat, a_feat], dim=-1)     # 拼接状态和动作特征
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        """保存模型权重到文件"""
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """从文件加载模型权重"""
        self.load_state_dict(_safe_load_state_dict(checkpoint_file))


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck 噪声过程
    
    生成时间相关的探索噪声，相比独立高斯噪声具有以下优势：
    1. 时间相关性：相邻时刻的噪声值相关，更适合物理系统的连续控制
    2. 均值回归：噪声值会向均值(0)回归，避免长期偏离
    3. 自适应衰减：支持随训练进展自动降低噪声强度
    """
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma_max=0.3, sigma_min=0.05, decay_rate=0.9995):
        """
        Args:
            size: 噪声维度（动作空间维度）
            mu: 均值回归目标（通常为0）
            theta: 均值回归速度（越大回归越快）
            sigma_max: 初始噪声强度（训练初期）
            sigma_min: 最小噪声强度（训练后期下限）
            decay_rate: 噪声衰减率（每步乘以该值）
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma_max
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate
        self.state = np.ones(self.size) * self.mu
        
    def reset(self):
        """重置噪声状态（每个episode开始时调用）"""
        self.state = np.ones(self.size) * self.mu
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)
        
    def decay_noise(self):
        """衰减噪声强度（每个episode结束时调用）"""
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)
        
    def sample(self):
        """
        采样OU噪声
        
        Returns:
            np.ndarray: 噪声向量
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state.copy()


class TD3:
    """
    双延迟深度确定性策略梯度（TD3）算法实现
    
    改进特性：
    1. 双 Actor 网络 + 软投票集成：提高决策鲁棒性
    2. 风险感知转向增强：靠近障碍物时自动增大转向幅度
    3. 共识正则化：鼓励两个 Actor 输出一致的动作
    4. Huber Loss：对异常奖励值更鲁棒
    5. 梯度裁剪：防止梯度爆炸
    """
    
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, action_limit_v, action_limit_w,
                 gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=1000000,
                 batch_size=512, cuda_empty_cache_interval=0, use_per=False, per_alpha=0.6):
        """
        Args:
            alpha: Actor 学习率
            beta: Critic 学习率
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            actor_fc1_dim: Actor 第一层 FC 维度
            actor_fc2_dim: Actor 第二层 FC 维度
            critic_fc1_dim: Critic 编码器融合维度
            critic_fc2_dim: Critic 第二层 FC 维度
            ckpt_dir: 模型保存路径
            action_limit_v: 线速度上限
            action_limit_w: 角速度上限
            gamma: 折扣因子
            tau: 软更新系数
            action_noise: 探索噪声标准差
            policy_noise: 策略噪声标准差（用于目标策略平滑）
            policy_noise_clip: 策略噪声裁剪范围
            delay_time: 延迟更新间隔（每隔 delay_time 步更新一次 Actor）
            max_size: 经验回放缓冲区最大容量
            batch_size: 训练批量大小
            cuda_empty_cache_interval: CUDA 显存清理间隔（0 表示不清理）
        """
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir
        self.start_epoch = 0
        self.bath_size = batch_size
        self.cuda_empty_cache_interval = max(0, int(cuda_empty_cache_interval))

        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.exploration_noise = OrnsteinUhlenbeckNoise(
            size=action_dim,
            mu=0.0,
            theta=0.20,
            sigma_max=0.35,
            sigma_min=0.08,
            decay_rate=0.9992,
        )

        # 状态组成：[激光雷达扇区数据, heading(1维), distance(1维)]
        self.goal_dim = 2
        self.scan_dim = state_dim - self.goal_dim
        if self.scan_dim < 4:
            raise ValueError("state_dim is too small for lidar+goal representation")

        # 投票集成超参数
        self.vote_temp = 0.35          # softmax 温度系数
        self.vote_risk_range = 0.80    # 风险感知范围上限
        self.vote_collision_range = 0.20  # 碰撞判定范围
        self.angular_vote_boost = 0.25  # 角速度增强系数（软调整模式下降低）

        # 创建网络实例
        self.actor_a = ActorNetwork(
            alpha=alpha, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,
            action_limit_v=action_limit_v, action_limit_w=action_limit_w,
        )
        self.actor_b = ActorNetwork(
            alpha=alpha, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,
            action_limit_v=action_limit_v, action_limit_w=action_limit_w,
        )

        self.critic1 = CriticNetwork(
            beta=beta, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim,
        )
        self.critic2 = CriticNetwork(
            beta=beta, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim,
        )

        self.target_actor_a = ActorNetwork(
            alpha=alpha, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,
            action_limit_v=action_limit_v, action_limit_w=action_limit_w,
        )
        self.target_actor_b = ActorNetwork(
            alpha=alpha, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,
            action_limit_v=action_limit_v, action_limit_w=action_limit_w,
        )
        self.target_critic1 = CriticNetwork(
            beta=beta, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim,
        )
        self.target_critic2 = CriticNetwork(
            beta=beta, scan_dim=self.scan_dim, goal_dim=self.goal_dim, action_dim=action_dim,
            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim,
        )
 
        # 经验回放缓冲区
        if use_per:
            self.memory = PrioritizedReplayBuffer(
                max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                batch_size=batch_size, alpha=per_alpha
            )
            self.use_per = True
        else:
            self.memory = ReplayBuffer(
                max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                batch_size=batch_size
            )
            self.use_per = False
        # 初始化目标网络参数
        self.update_network_parameters(tau=1.0)


    def update_network_parameters(self, tau=None):
        """
        软更新目标网络参数（EMA 更新）
        
        θ_target = τ * θ_local + (1 - τ) * θ_target
        
        Args:
            tau: 软更新系数（默认使用 self.tau）
        """
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor_a.parameters(), self.target_actor_a.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for actor_params, target_actor_params in zip(self.actor_b.parameters(), self.target_actor_b.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        """存储经验到回放缓冲区"""
        self.memory.store_transition(state, action, reward, state_, done)

    def _vote_actions(self, state_tensor, use_target=False):
        """
        双 Actor 投票集成决策
        
        核心流程：
        1. 两个 Actor 分别输出动作
        2. 用两个 Critic 评估 Q 值，取最小值
        3. Softmax 加权投票
        4. 风险感知转向增强：靠近障碍物时增大转向幅度
        
        Args:
            state_tensor: 状态张量
            use_target: 是否使用目标网络（训练时为 True，推理时为 False）
            
        Returns:
            tuple: (voted_action, action_a, action_b, vote_weight)
        """
        if use_target:
            action_a = self.target_actor_a.forward(state_tensor)
            action_b = self.target_actor_b.forward(state_tensor)
        else:
            action_a = self.actor_a.forward(state_tensor)
            action_b = self.actor_b.forward(state_tensor)

        # 使用双 Critic 最小 Q 值作为评分（避免高估）
        with T.no_grad():
            q_a = T.min(
                self.critic1.forward(state_tensor, action_a),
                self.critic2.forward(state_tensor, action_a),
            )
            q_b = T.min(
                self.critic1.forward(state_tensor, action_b),
                self.critic2.forward(state_tensor, action_b),
            )

        # Softmax 投票：Q 值越高，权重越大
        score = T.cat([q_a, q_b], dim=1) / self.vote_temp
        vote_weight = F.softmax(score, dim=1)
        voted_action = vote_weight[:, 0:1] * action_a + vote_weight[:, 1:2] * action_b

        # --- 风险感知转向增强（软调整）---
        # 计算前方扇区风险系数（避免侧后方影响）
        scan = state_tensor[:, :self.scan_dim]
        scan_center = self.scan_dim // 2
        front_half_width = max(1, self.scan_dim // 6)
        front_start = scan_center - front_half_width
        front_end = scan_center + front_half_width + 1
        front_scan = scan[:, front_start:front_end]
        front_min = T.min(front_scan, dim=1, keepdim=True).values

        # 计算风险系数 [0, 1]
        risk = T.clamp(
            (self.vote_risk_range - front_min) / (self.vote_risk_range - self.vote_collision_range + 1e-6),
            0.0,
            1.0,
        )

        # 使用heading信息确定转向方向（从状态中提取）
        goal_heading = state_tensor[:, self.scan_dim:self.scan_dim + 1]
        turn_direction = T.sign(goal_heading)

        # 软增强：仅在风险较高时适当增大转向幅度
        # 使用加权混合而非覆盖，避免强制注入转向偏置
        voted_w = voted_action[:, 1:2]
        # 计算建议的最小转向（随风险线性增加）
        suggested_turn = self.angular_vote_boost * risk * T.tanh(risk * 3.0)

        # 混合策略：风险低时保持原动作，风险高时适度增加转向
        # 使用 risk^2 使增强更平滑，避免低风险时过度干预
        risk_weight = risk ** 2
        enhanced_w = T.sign(voted_w + 1e-6) * T.maximum(T.abs(voted_w), suggested_turn)

        voted_w = voted_w + risk_weight * (enhanced_w - voted_w)
        voted_w = T.where(
            T.abs(voted_w) < 0.02,
            T.zeros_like(voted_w),
            voted_w,
        )
        voted_action[:, 1:2] = T.clamp(voted_w, -self.action_limit_w, self.action_limit_w)
        voted_action[:, 0:1] = T.clamp(voted_action[:, 0:1], 0.0, self.action_limit_v)

        return voted_action, action_a, action_b, vote_weight

    def choose_action(self, observation, train=True):
        """
        选择动作
        
        Args:
            observation: 观测状态
            train: 是否处于训练模式（True 时添加探索噪声）
            
        Returns:
            np.ndarray: 动作向量
        """
        self.actor_a.eval()
        self.actor_b.eval()
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observation = np.nan_to_num(observation, nan=0.0, posinf=self.action_limit_v, neginf=0.0)
        state = T.from_numpy(observation).to(device)
        with T.no_grad():
            action, _, _, _ = self._vote_actions(state, use_target=False)
            action = T.nan_to_num(action, nan=0.0, posinf=self.action_limit_v, neginf=0.0)

            if train:
                ou_noise = self.exploration_noise.sample()
                noise_scale = T.tensor([0.50, 1.20], dtype=T.float32, device=device)
                noise = T.tensor(ou_noise * noise_scale.cpu().numpy(), dtype=T.float32, device=device).unsqueeze(0)
                action = action + noise
                action[0][0] = T.clamp(action[0][0], 0.0, self.action_limit_v)
                action[0][1] = T.clamp(action[0][1], -self.action_limit_w, self.action_limit_w)

            action = T.nan_to_num(action, nan=0.0, posinf=self.action_limit_v, neginf=0.0)
        self.actor_a.train()
        self.actor_b.train()
        return action.squeeze().cpu().numpy()

    def learn(self):
        """
        执行一次 TD3 训练步骤
        
        训练流程：
        1. 采样经验数据
        2. 计算目标 Q 值（目标策略平滑 + 噪声）
        3. 更新 Critic1 和 Critic2
        4. 延迟更新 Actor_a 和 Actor_b（含共识正则化）
        5. 软更新目标网络
        
        Returns:
            dict or None: 训练损失字典，包含 critic1_loss, critic2_loss, actor_a_loss, actor_b_loss
                          如果未执行Actor更新（延迟策略），则只返回Critic损失
                          如果缓冲区未准备好，返回None
        """
        if not self.memory.ready():
            return None

        # 采样并清洗数据
        if self.use_per:
            states, actions, rewards, states_, terminals, indices, weights = self.memory.sample_buffer()
            weights_tensor = T.as_tensor(weights, dtype=T.float32, device=device)
        else:
            states, actions, rewards, states_, terminals = self.memory.sample_buffer()
            weights_tensor = T.ones_like(T.as_tensor(rewards, dtype=T.float32, device=device))
            
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=self.action_limit_v, neginf=-self.action_limit_w)
        rewards = np.nan_to_num(rewards, nan=-200.0, posinf=200.0, neginf=-200.0)
        states_ = np.nan_to_num(states_, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为 PyTorch 张量
        states_tensor = T.as_tensor(states, dtype=T.float32, device=device)
        actions_tensor = T.as_tensor(actions, dtype=T.float32, device=device)
        rewards_tensor = T.as_tensor(rewards, dtype=T.float32, device=device)
        next_states_tensor = T.as_tensor(states_, dtype=T.float32, device=device)
        terminals_tensor = T.as_tensor(terminals, dtype=T.bool, device=device)

        # --- 计算目标 Q 值 ---
        with T.no_grad():
            # 使用目标网络计算下一动作（投票集成）
            next_actions_tensor, _, _, _ = self._vote_actions(next_states_tensor, use_target=True)
            # 添加目标策略噪声（策略平滑）
            action_noise = T.tensor(
                np.stack([
                    np.random.normal(loc=0.0, scale=self.policy_noise * 0.6, size=next_actions_tensor.shape[0]),
                    np.random.normal(loc=0.0, scale=self.policy_noise * 1.4, size=next_actions_tensor.shape[0]),
                ], axis=1),
                dtype=T.float,
            ).to(device)

            # 噪声裁剪（TD3 策略平滑）
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = next_actions_tensor + action_noise
            next_actions_tensor[:, 0] = T.clamp(next_actions_tensor[:, 0], 0.0, self.action_limit_v)
            next_actions_tensor[:, 1] = T.clamp(next_actions_tensor[:, 1], -self.action_limit_w, self.action_limit_w)
            
            # 计算目标 Q 值（取双 Critic 最小值）
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0  # 终止状态 Q 值为 0
            q2_[terminals_tensor] = 0.0
            critic_val = T.min(q1_, q2_)
            # TD 目标：r + γ * min(Q1', Q2')
            target = rewards_tensor + self.gamma * critic_val

        # --- 更新 Critic 网络 ---
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        # 使用 Huber Loss（对异常值更鲁棒）
        if self.use_per:
            critic1_loss = T.mean(weights_tensor * F.huber_loss(q1, target.detach(), reduction='none'))
            critic2_loss = T.mean(weights_tensor * F.huber_loss(q2, target.detach(), reduction='none'))
        else:
            critic1_loss = F.huber_loss(q1, target.detach())
            critic2_loss = F.huber_loss(q2, target.detach())

        # Critic1 反向传播 + 梯度裁剪
        self.critic1.optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=5.0)
        self.critic1.optimizer.step()

        # Critic2 反向传播 + 梯度裁剪
        self.critic2.optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=5.0)
        self.critic2.optimizer.step()

        self.update_time += 1
        
        # 记录Critic损失
        loss_dict = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
        }
        
        # 延迟更新：每隔 delay_time 步更新一次 Actor
        if self.update_time % self.delay_time != 0:
            return loss_dict

        # --- 延迟策略更新（双 Actor + 共识正则化）---
        # 冻结 Critic 参数，避免无效累积梯度
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # 更新 Actor_a
        self.actor_a.optimizer.zero_grad(set_to_none=True)
        act_a = self.actor_a.forward(states_tensor)
        with T.no_grad():
            act_b_ref = self.actor_b.forward(states_tensor)
        qa = T.min(self.critic1.forward(states_tensor, act_a), self.critic2.forward(states_tensor, act_a))
        vote_consensus_a = T.mean((act_a - act_b_ref) ** 2)  # 共识惩罚项
        actor_a_loss = -T.mean(qa) + 0.01 * vote_consensus_a  # 最大化 Q 值 + 最小化分歧
        actor_a_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor_a.parameters(), max_norm=5.0)
        self.actor_a.optimizer.step()

        # 更新 Actor_b
        self.actor_b.optimizer.zero_grad(set_to_none=True)
        act_b = self.actor_b.forward(states_tensor)
        with T.no_grad():
            act_a_ref = self.actor_a.forward(states_tensor)
        qb = T.min(self.critic1.forward(states_tensor, act_b), self.critic2.forward(states_tensor, act_b))
        vote_consensus_b = T.mean((act_b - act_a_ref) ** 2)  # 共识惩罚项
        actor_b_loss = -T.mean(qb) + 0.01 * vote_consensus_b
        actor_b_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor_b.parameters(), max_norm=5.0)
        self.actor_b.optimizer.step()

        # 恢复 Critic 参数可训练状态
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        if self.use_per:
            with T.no_grad():
                td_errors = T.abs(q1 - target) + T.abs(q2 - target)
                td_errors_np = td_errors.cpu().numpy()
                self.memory.update_priorities(indices, td_errors_np)

        # 软更新目标网络
        self.update_network_parameters()
        
        # 记录Actor损失
        loss_dict['actor_a_loss'] = actor_a_loss.item()
        loss_dict['actor_b_loss'] = actor_b_loss.item()
        
        # CUDA 显存清理（可选）
        if self.cuda_empty_cache_interval > 0 and T.cuda.is_available():
            if self.update_time % self.cuda_empty_cache_interval == 0:
                T.cuda.empty_cache()
        
        return loss_dict

    def reset_exploration_noise(self):
        """重置探索噪声状态（每个episode开始时调用）"""
        self.exploration_noise.reset()
        
    def save_models(self, episode):
        """
        保存所有网络权重（分文件夹存储）
        
        存储结构：
            checkpoint_dir/
            ├── actor/          # Actor-A策略网络
            ├── actor_b/        # Actor-B策略网络
            ├── critic1/        # Critic1价值网络
            ├── critic2/        # Critic2价值网络
            └── target/         # 目标网络
        
        Args:
            episode: 当前训练轮次编号
        """
        import os
        
        # 创建子文件夹
        actor_dir = os.path.join(self.checkpoint_dir, 'actor')
        actor_b_dir = os.path.join(self.checkpoint_dir, 'actor_b')
        critic1_dir = os.path.join(self.checkpoint_dir, 'critic1')
        critic2_dir = os.path.join(self.checkpoint_dir, 'critic2')
        target_dir = os.path.join(self.checkpoint_dir, 'target')
        
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(actor_b_dir, exist_ok=True)
        os.makedirs(critic1_dir, exist_ok=True)
        os.makedirs(critic2_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        # 保存Actor网络
        self.actor_a.save_checkpoint(os.path.join(actor_dir, 'actor_{}.pth'.format(episode)))
        self.actor_b.save_checkpoint(os.path.join(actor_b_dir, 'actor_b_{}.pth'.format(episode)))
        
        # 保存Critic网络（分开存储）
        self.critic1.save_checkpoint(os.path.join(critic1_dir, 'critic1_{}.pth'.format(episode)))
        self.critic2.save_checkpoint(os.path.join(critic2_dir, 'critic2_{}.pth'.format(episode)))
        
        # 保存目标网络
        self.target_actor_a.save_checkpoint(os.path.join(target_dir, 'target_actor_{}.pth'.format(episode)))
        self.target_actor_b.save_checkpoint(os.path.join(target_dir, 'target_actor_b_{}.pth'.format(episode)))
        self.target_critic1.save_checkpoint(os.path.join(target_dir, 'target_critic1_{}.pth'.format(episode)))
        self.target_critic2.save_checkpoint(os.path.join(target_dir, 'target_critic2_{}.pth'.format(episode)))
        
        print('=== Models saved at episode {} ==='.format(episode))
        print('  Actor: {}'.format(actor_dir))
        print('  Critic1: {}'.format(critic1_dir))
        print('  Critic2: {}'.format(critic2_dir))
        print('  Target: {}'.format(target_dir))

    def load_models(self, episode):
        """
        加载所有网络权重
        
        加载结构（子文件夹）：
            actor/actor_{episode}.pth
            target/target_actor_{episode}.pth
            actor_b/actor_b_{episode}.pth  (可选)
            target/target_actor_b_{episode}.pth  (可选)
            critic1/critic1_{episode}.pth
            target/target_critic1_{episode}.pth
            critic2/critic2_{episode}.pth
            target/target_critic2_{episode}.pth
        
        支持向后兼容：如果 actor_b 的权重文件不存在，则从 actor_a 复制。
        
        Args:
            episode: 要加载的轮次编号
        """
        # 加载 Actor-A
        actor_a_file = os.path.join(self.checkpoint_dir, 'actor', 'actor_{}.pth'.format(episode))
        self.actor_a.load_checkpoint(actor_a_file)
        print('Loading actor_a network successfully!')
        
        target_actor_a_file = os.path.join(self.checkpoint_dir, 'target', 'target_actor_{}.pth'.format(episode))
        self.target_actor_a.load_checkpoint(target_actor_a_file)
        print('Loading target_actor_a network successfully!')

        # 加载 Actor-B（可选，向后兼容）
        actor_b_file = os.path.join(self.checkpoint_dir, 'actor_b', 'actor_b_{}.pth'.format(episode))
        target_actor_b_file = os.path.join(self.checkpoint_dir, 'target', 'target_actor_b_{}.pth'.format(episode))
        if os.path.isfile(actor_b_file) and os.path.isfile(target_actor_b_file):
            self.actor_b.load_checkpoint(actor_b_file)
            self.target_actor_b.load_checkpoint(target_actor_b_file)
            print('Loading actor_b and target_actor_b network successfully!')
        else:
            # 向后兼容：旧版本检查点没有 actor_b
            self.actor_b.load_state_dict(self.actor_a.state_dict())
            self.target_actor_b.load_state_dict(self.target_actor_a.state_dict())
            print('No actor_b checkpoint found, initialized actor_b from actor_a.')

        # 加载 Critic1
        critic1_file = os.path.join(self.checkpoint_dir, 'critic1', 'critic1_{}.pth'.format(episode))
        self.critic1.load_checkpoint(critic1_file)
        print('Loading critic1 network successfully!')
        
        target_critic1_file = os.path.join(self.checkpoint_dir, 'target', 'target_critic1_{}.pth'.format(episode))
        self.target_critic1.load_checkpoint(target_critic1_file)
        print('Loading target critic1 network successfully!')
        
        # 加载 Critic2
        critic2_file = os.path.join(self.checkpoint_dir, 'critic2', 'critic2_{}.pth'.format(episode))
        self.critic2.load_checkpoint(critic2_file)
        print('Loading critic2 network successfully!')
        
        target_critic2_file = os.path.join(self.checkpoint_dir, 'target', 'target_critic2_{}.pth'.format(episode))
        self.target_critic2.load_checkpoint(target_critic2_file)
        print('Loading target critic2 network successfully!')

    def load_pretrained_weights(self, pretrained_dir, pretrained_episode):
        """
        加载预训练权重用于微调（课程学习支持）
        
        核心机制：
        1. 只加载主网络（actor_a, actor_b, critic1, critic2）的参数
        2. 目标网络通过软更新同步（tau=1.0 硬拷贝）
        3. 如果预训练权重中没有 actor_b，则从 actor_a 复制
        
        适用场景：
        - 课程学习：加载上一阶段训练的权重作为当前阶段起点
        - 迁移学习：加载相似任务的预训练模型进行微调
        - 恢复训练：从中断的检查点继续训练
        
        Args:
            pretrained_dir: 预训练模型所在目录（可能与当前 ckpt_dir 不同）
            pretrained_episode: 预训练模型的轮次编号
        """
        actor_a_file = pretrained_dir + '/TD3_actor_{}.pth'.format(pretrained_episode)
        if not os.path.isfile(actor_a_file):
            raise FileNotFoundError(f"预训练模型不存在: {actor_a_file}")

        self.actor_a.load_checkpoint(actor_a_file)
        print(f'Loading pretrained actor_a from {actor_a_file} successfully!')

        actor_b_file = pretrained_dir + '/TD3_actor_b_{}.pth'.format(pretrained_episode)
        if os.path.isfile(actor_b_file):
            self.actor_b.load_checkpoint(actor_b_file)
            print(f'Loading pretrained actor_b from {actor_b_file} successfully!')
        else:
            self.actor_b.load_state_dict(self.actor_a.state_dict())
            print('No pretrained actor_b found, initialized actor_b from actor_a.')

        critic1_file = pretrained_dir + '/TD3_critic1_{}.pth'.format(pretrained_episode)
        if os.path.isfile(critic1_file):
            self.critic1.load_checkpoint(critic1_file)
            print(f'Loading pretrained critic1 from {critic1_file} successfully!')
        else:
            print(f'Warning: No pretrained critic1 found at {critic1_file}')

        critic2_file = pretrained_dir + '/TD3_critic2_{}.pth'.format(pretrained_episode)
        if os.path.isfile(critic2_file):
            self.critic2.load_checkpoint(critic2_file)
            print(f'Loading pretrained critic2 from {critic2_file} successfully!')
        else:
            print(f'Warning: No pretrained critic2 found at {critic2_file}')

        self.update_network_parameters(tau=1.0)
        print('Target networks synchronized with pretrained weights (tau=1.0).')
