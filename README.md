
https://github.com/user-attachments/assets/6b18b180-5952-46c2-a1e0-15915dc91f4d

# 基于RAV-TD3的无人车自主导航与动态避障系统

> 基于 RAV-TD3 (Risk-Aware Voting TD3) 算法的端到端无人车导航解决方案

> 注意：goal_box的gazebo模型可视化有bug，可视化的goal与真实的goal坐标偶尔会出现不同步的情况，以终端打印的goal坐标为准。

> 持续更新中，算法持续优化中
---

## 目录

1. [项目概述](#1-项目概述)
2. [算法原理](#2-算法原理)
3. [系统架构](#3-系统架构)
4. [状态空间与动作空间](#4-状态空间与动作空间)
5. [奖励函数设计](#5-奖励函数设计)
6. [网络架构](#6-网络架构)
7. [项目文件结构](#7-项目文件结构)
8. [环境配置与依赖](#8-环境配置与依赖)
9. [运行步骤](#9-运行步骤)
10. [关键代码解读](#10-关键代码解读)
11. [实验结果与分析](#11-实验结果与分析)
12. [总结与展望](#12-总结与展望)
13. [参考文献](#13-参考文献)
14. [模型与地图说明](#14-模型与地图说明)
15. [附录：快速开始命令速查](#15-附录快速开始命令速查)
16. [开源协议](#16-开源协议)
17. [致谢](#17-致谢)

---
## 1. 项目概述

### 1.1 项目背景

本项目针对复杂机场环境下无人转运车的自主导航与动态避障需求，基于**深度强化学习（Deep Reinforcement Learning, DRL）**技术，构建了一套端到端的自主导航系统（RAV-TD3算法）。系统以激光雷达感知信息与目标点极坐标为输入，通过 Actor-Critic 架构的智能体输出线速度与角速度控制指令，实现无人车在机库、停机坪、跑道等复杂机场环境下的自主决策与安全导航。算法效果如下：



https://github.com/user-attachments/assets/e114ef59-9ea1-4ede-84ca-bb4d2a75b906




### 1.2 技术栈

| 组件 | 技术/框架 | 用途 |
|------|----------|------|
| 机器人中间件 | ROS Noetic | 节点通信与传感器接入 |
| 物理仿真 | Gazebo 11 | 高保真环境仿真 |
| 深度学习 | PyTorch | 神经网络训练与推理 |
| 强化学习算法 | RAV-TD3 | 连续动作空间策略学习 |
| 编程语言 | Python 3.8+ / C++ | 算法实现与ROS节点 |

### 1.3 核心特性

- **端到端学习**：直接从传感器数据到控制指令的映射
- **双策略投票机制**：Actor-A 和 Actor-B 双网络投票决策
- **风险感知注意力**：激光雷达扇区注意力编码器
- **多目标奖励函数**：平衡导航效率、避障安全与运动平稳性
- **自适应OU探索噪声**：时间相关的连续控制噪声，支持自动衰减
- **优先经验回放(PER)**：基于TD-error的优先级采样，加速重要经验学习
- **课程学习支持**：加载预训练权重进行微调，支持多阶段训练

---

## 2. 算法原理

### 2.1 RAV-TD3 总体框架

**RAV-TD3（Risk-Aware Voting TD3）** 是在 TD3 算法基础上，针对复杂环境下自主导航的**安全性、鲁棒性和决策多样性**需求，提出的融合五项核心创新的强化学习算法。

$$\text{RAV-TD3} = \underbrace{\text{TD3}}_{\text{基础框架}} + \underbrace{\text{双策略投票} + \text{风险感知注意力} + \text{风险感知奖励} + \text{自适应OU噪声} + \text{优先经验回放}}_{\text{五项核心创新}}$$

| 编号 | 创新模块 | 问题痛点 | 核心思想 |
|:---:|---------|---------|---------|
| ① | **双策略投票机制** | 单策略决策鲁棒性不足，复杂场景易陷入局部最优 | 两个独立 Actor 通过 Critic Q 值 Softmax 软投票，高风险时增强转向 |
| ② | **极坐标注意力编码器** | 全连接处理激光雷达无法区分不同方向的重要性 | 可学习注意力 + 距离逆偏置，自适应聚焦最近障碍物扇区 |
| ③ | **风险感知多目标奖励** | 固定权重奖励无法根据障碍物距离动态调整策略 | 引入风险系数 $risk$ 动态调节各奖励项权重，安全激进/危险保守 |
| ④ | **自适应 OU 探索噪声** | 固定高斯噪声时间无关，不适合物理系统连续控制 | OU 过程提供时序相关噪声 + 各向异性缩放 + 自动衰减 |
| ⑤ | **优先经验回放(PER)** | 均匀采样浪费高价值经验（碰撞、到达目标） | 基于 TD-error 优先级采样 + 重要性权重偏差校正 |

> 以下 2.2 节简要回顾 RAV-TD3 继承的 TD3 基础框架，2.3~2.7 节逐一详述五项创新。

### 2.2 TD3 基础回顾

RAV-TD3 继承了 TD3 解决 DDPG 值函数过估计问题的三大核心机制：

#### 2.2.1 双重 Q 网络（Clipped Double Q-Learning）

使用两个独立 Critic 网络 $Q_{\theta_1}$、$Q_{\theta_2}$，目标值取二者较小值，抑制乐观估计：

$$y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', a')$$

#### 2.2.2 目标策略平滑（Target Policy Smoothing）

对目标动作添加裁剪噪声，平滑 Q 值估计，减少策略更新方差：

$$a' = \pi_{\phi}(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

本项目设置：训练 $\sigma = 0.25, c = 0.35$；测试 $\sigma = 0.08, c = 0.3$。

#### 2.2.3 延迟策略更新与软更新

Actor 更新频率低于 Critic（$d = 2$，每两步 Critic 更新一次 Actor），策略梯度：

$$\phi \leftarrow \nabla_\phi \mathbb{E}[-Q_{\theta_1}(s, \pi_\phi(s))]$$

目标网络软更新：$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$，$\tau = 0.005$。

贝尔曼方程基础（$\gamma = 0.99$）：

$$Q^\pi(s, a) = \mathbb{E}\left[r(s, a) + \gamma \mathbb{E}_{a' \sim \pi(s')}[Q^\pi(s', a')]\right]$$

---

### 2.3 创新一：双策略投票机制（Dual-Actor Voting）

#### 2.3.1 动机

标准 TD3 使用单个 Actor 策略网络，在复杂避障场景中容易产生决策偏差（如偏好某一侧转向，或在两个可行方向间抖动）。RAV-TD3 引入两个独立初始化的 Actor 网络，通过 **Critic 评估 + Softmax 软投票** 融合多种决策建议，显著提升鲁棒性。

#### 2.3.2 双 Actor Softmax 投票公式

两个 Actor 各自输出动作，由双重 Critic 评估后软投票：

$$a_A = \pi_A(s), \quad a_B = \pi_B(s)$$

$$Q_A = \min(Q_{\theta_1}(s, a_A),\ Q_{\theta_2}(s, a_A))$$

$$Q_B = \min(Q_{\theta_1}(s, a_B),\ Q_{\theta_2}(s, a_B))$$

$$[w_A, w_B] = \text{softmax}\left(\frac{[Q_A, Q_B]}{T}\right)$$

$$a_{voted} = w_A \cdot a_A + w_B \cdot a_B$$

其中 $T$ 为投票温度系数，控制软投票的尖锐程度。Q 值高的动作获得更高权重。

#### 2.3.3 风险感知角速度增强

障碍物临近时，在投票动作基础上进一步增强转向幅度：

$$risk = \text{clip}\left(\frac{d_{risk} - d_{front}}{d_{risk} - d_{collision}},\ 0,\ 1\right) = \text{clip}\left(\frac{0.85 - d_{front}}{0.85 - 0.18},\ 0,\ 1\right)$$

$$a_{enhanced}[\omega] = a_{voted}[\omega] + risk^2 \cdot (a_{stronger}[\omega] - a_{voted}[\omega])$$

其中 $a_{stronger}$ 取 $\{a_A, a_B\}$ 中 $|\omega|$ 更大的动作。采用 $risk^2$ 平滑过渡：
- $risk \to 0$（安全）：$risk^2 \to 0$，保留网络原始意图
- $risk \to 1$（紧迫）：$risk^2 \to 1$，最大程度增强避障转向

#### 2.3.4 共识正则化（Consensus Regularization）

Actor 更新时引入一致性约束，促进两个策略在训练中协调收敛（$\lambda_{cons} = 0.01$）：

$$\mathcal{L}_{actor}^A = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q_{\theta_1}(s, \pi_A(s))\right] + \lambda_{cons} \cdot \|\pi_A(s) - \pi_B(s)\|^2$$

$$\mathcal{L}_{actor}^B = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q_{\theta_1}(s, \pi_B(s))\right] + \lambda_{cons} \cdot \|\pi_B(s) - \pi_A(s)\|^2$$

对方动作均 `.detach()` 截断梯度，保障两个 Actor 独立学习的同时趋于一致。

---

### 2.4 创新二：极坐标激光雷达注意力编码器（PolarLidarAttentionEncoder）

#### 2.4.1 动机

传统方法将 360° 激光雷达数据直接输入全连接层，无法显式建模"哪个方向的障碍物更重要"。RAV-TD3 设计了融合**可学习注意力**和**距离逆风险偏置**的极坐标编码器。

#### 2.4.2 扇区嵌入

将 36 个激光雷达扇区的距离值通过线性层投影到 48 维：

$$e_i = \text{ReLU}\left(\text{LayerNorm}(W_s \cdot d_i + b_s)\right) \in \mathbb{R}^{48},\quad i = 1,\dots,36$$

#### 2.4.3 注意力权重：可学习 + 风险偏置

$$\alpha_i = \text{softmax}\left(\underbrace{W_a \cdot e_i}_{\text{可学习注意力 logit}} + \underbrace{\frac{\beta}{d_i + 0.05}}_{\text{显式风险偏置}}\right)$$

- **可学习注意力** $W_a \cdot e_i$：网络自动学习各扇区的导航重要性
- **风险偏置** $\beta/(d_i + 0.05)$（$\beta = 1.5$）：显式放大近距离扇区的注意力，即使未经训练也会优先关注障碍物方向

#### 2.4.4 上下文向量与多源融合

$$\mathbf{c} = \sum_{i=1}^{36} \alpha_i \cdot e_i \in \mathbb{R}^{48}$$

最终将注意力上下文、目标点 MLP 特征（$\mathbf{g} \in \mathbb{R}^{64}$）和激光雷达统计特征拼接融合：

$$\mathbf{h} = \text{ReLU}\left(\text{LayerNorm}\left(W_f \cdot [\mathbf{c}; \mathbf{g}; s_{min}; s_{mean}; s_{std}] + b_f\right)\right) \in \mathbb{R}^{192}$$

$\mathbf{h}$ 作为 Actor 和 Critic 网络的共享输入编码。

> 详细网络架构见 [第 6 节：网络架构](#6-网络架构)。

---

### 2.5 创新三：风险感知多目标奖励函数

#### 2.5.1 风险系数定义

基于前方激光雷达最小距离 $d_{front}$ 定义归一化风险：

$$risk = \text{clip}\left(\frac{0.85 - d_{front}}{0.85 - 0.18},\ 0,\ 1\right)$$

- $d_{front} \geq 0.85m$ → $risk = 0$（安全开阔区）
- $d_{front} \to 0.18m$ → $risk \to 1$（碰撞临界区）

#### 2.5.2 风险感知奖励项

总奖励 $r_{total} = \sum w_i \cdot r_i$，核心奖励项的风险感知设计：

| 奖励项 | 公式 | 权重 | 风险感知机制 |
|-------|------|:---:|------------|
| 目标推进 | $r_{goal} = \Delta d \cdot (1 - 0.35 \cdot risk)$ | 22.0 | 高风险自动降低前进激励 |
| 速度奖励 | $r_{speed} = v \cdot (1 - 0.5 \cdot risk)$ | 1.0 | 高风险抑制高速行驶 |
| 转向鼓励 | $r_{turn} = \|\omega\| \cdot (0.25 + risk)$ | 1.2 | 高风险自动增强转向激励 |
| 风险直行惩罚 | $r_{straight} = -risk \cdot \max(0.22 - \|\omega\|, 0)$ | 2.5 | 高风险惩罚直行行为 |
| 风险高速惩罚 | $r_{fast} = -risk \cdot \max(v - 0.25, 0)$ | 2.0 | 高风险惩罚高速行驶 |

非风险相关辅助项：

| 奖励项 | 公式 | 权重 | 说明 |
|-------|------|:---:|------|
| 朝向奖励 | $r_{heading} = \cos(heading)$ | 1.6 | 鼓励朝向目标点 |
| 安全间隙 | $r_{clear} = \tanh((d_{front} - 0.18) \times 3)$ | 2.0 | 鼓励远离障碍物 |
| 平滑惩罚 | $r_{smooth} = -(\|\Delta v\| + 0.8 \|\Delta \omega\|)$ | 0.25 | 抑制动作突变 |
| 存活奖励 | $r_{survive} = 1$ | 0.10 | 鼓励持续探索 |

#### 2.5.3 终止条件

| 触发条件 | 奖励 | 含义 |
|---------|:---:|------|
| $d_{min} < 0.18m$ | **-260** | 碰撞终止 |
| $d_{goal} < 0.35m$ | **+420** | 成功到达 |
| 连续 120 步 $v < 0.05$ 且 $\|\omega\| > 0.6$ | -80 | 旋转卡死 |
| 连续 140 步未向目标靠近 | -100 | 无进度卡死 |
| 步数达到 1000 | 0 | 超时终止 |

> 完整奖励实现代码见 [第 5 节：奖励函数设计](#5-奖励函数设计)。

---

### 2.6 创新四：自适应 Ornstein-Uhlenbeck 探索噪声

#### 2.6.1 OU 噪声过程

Ornstein-Uhlenbeck 过程具有**时序相关性**和**均值回归**特性，相比独立高斯噪声更适合物理系统的连续控制：

$$x_{t+1} = x_t + \theta \cdot (\mu - x_t) + \sigma_t \cdot \mathcal{N}(0, 1)$$

| 参数 | 值 | 含义 |
|------|:---:|------|
| $\mu$ | 0 | 均值回归目标 |
| $\theta$ | 0.20 | 回归速率 |
| $\sigma_{max}$ | 0.35 | 初始噪声强度 |
| $\sigma_{min}$ | 0.08 | 最小噪声下限 |
| $\gamma_{decay}$ | 0.9992 | 每 episode 衰减率 |

#### 2.6.2 自适应衰减

$$\sigma_t = \max(\sigma_{min},\ \sigma_{t-1} \cdot \gamma_{decay})$$

训练初期高噪声促进广泛探索，后期低噪声让策略收敛。每 episode 结束执行一次衰减。

#### 2.6.3 各向异性缩放

针对线速度和角速度的物理差异，采用不同的噪声缩放：

$$\eta = [x_v \cdot 0.50,\ x_\omega \cdot 1.20]$$

角速度噪声强度约为线速度的 2.4 倍，因为转向探索需求更频繁。

> 完整噪声实现见 [第 4.3 节](#43-探索噪声设计)。

---

### 2.7 创新五：优先经验回放（Prioritized Experience Replay, PER）

#### 2.7.1 优先级定义

基于双 Critic 的 TD-error 均值定义经验 $i$ 的优先级（$\alpha = 0.6$）：

$$p_i = \left(\frac{|\delta_i^{(1)}| + |\delta_i^{(2)}|}{2}\right)^\alpha + \epsilon$$

$$\delta_i^{(k)} = Q_{\theta_k}(s_i, a_i) - (r_i + \gamma \min_j Q_{\theta_j'}(s_i', a_i'))$$

#### 2.7.2 采样概率与重要性权重

$$P(i) = \frac{p_i}{\sum_k p_k}$$

非均匀采样引入偏差，通过重要性采样权重校正：

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta_t}$$

$\beta_t$ 从初始值 $\beta_0 = 0.4$ 线性递增至 1，最终完全补偿偏差：

$$\beta_t = 0.4 + 0.6 \cdot \frac{t}{T_{max}}$$

#### 2.7.3 加权 Critic 更新

$$\mathcal{L}_{critic}^{(k)} = \frac{1}{N}\sum_i w_i \cdot \text{HuberLoss}\left(Q_{\theta_k}(s_i, a_i),\ y_i\right)$$

采样后立即用最新 TD-error 更新经验优先级。采用 SumTree 数据结构，优先级查询和更新复杂度 $O(\log N)$。

> 完整 PER 实现见 [第 10.3 节](#103-critic-更新支持per)。

---

### 2.8 RAV-TD3 完整算法流程

```
初始化：
    两个独立 Actor 网络 π_A, π_B  ← TD3 不同：双 Actor
    对应 Target Actor 网络 π_A', π_B'
    双重 Critic 网络 Q_1, Q_2 和 Target Q_1', Q_2'
    PER SumTree 缓冲区 D，容量 200000  ← TD3 不同：优先经验回放
    OU 噪声状态 x_v = x_ω = 0          ← TD3 不同：自适应OU噪声
    全局步数计数器 update_time = 0
    β_t = 0.4, 最大优先级 = 1.0

for episode = 1 to M:
    重置环境，获取 s_0, past_action = [0, 0]

    for t = 1 to T:
        if t < 1500:                             # 随机探索阶段
            a = [uniform(0.15, v_max), uniform(-w_max, w_max)]
        else:
            # ★ 创新 ①：双 Actor 投票
            a_A = π_A(s), a_B = π_B(s)
            Q_A = min(Q_1(s,a_A), Q_2(s,a_A))
            Q_B = min(Q_1(s,a_B), Q_2(s,a_B))
            [w_A, w_B] = softmax([Q_A/T, Q_B/T])
            a_voted = w_A·a_A + w_B·a_B

            # ★ 创新 ①：风险感知角速度增强
            risk = clamp((0.85 - d_front) / (0.85 - 0.18), 0, 1)
            a_voted[ω] += risk² · (a_stronger[ω] - a_voted[ω])

            # ★ 创新 ④：OU 噪声 + 各向异性缩放
            a = a_voted + OU_noise · [0.50, 1.20]

        执行 a，获得 r, s', done

        # ★ 创新 ③：风险感知多目标奖励
        risk = clamp((0.85 - d_front) / (0.85 - 0.18), 0, 1)
        r = 22·Δd·(1-0.35·risk) + 1.6·cos(heading) + 2.0·tanh((d_front-0.18)·3)
            + 1.0·v·(1-0.5·risk) + 1.2·|ω|·(0.25+risk)
            - 0.25·(|Δv|+0.8·|Δω|) - 2.5·risk·max(0.22-|ω|,0)
            - 2.0·risk·max(v-0.25,0) + 0.10

        # ★ 创新 ⑤：优先经验回放存储
        D.store((s, a, r, s', done), max_priority)

        # ---------- Critic 更新 ----------
        从 D 按优先级采样 batch=256 条，获取 (s_i, a_i, r_i, s_i', done_i, idx_i, w_i)

        # ★ 创新 ①：目标动作也用双 Actor 投票
        a'_voted, _, _, _ = vote(π_A'(s'), π_B'(s'))
        a'_noisy = a'_voted + clip(N(0,σ), -c, c)

        y_i = r_i + γ · (1-done_i) · min(Q_1'(s_i', a'_noisy), Q_2'(s_i', a'_noisy))

        # ★ 创新 ⑤：加权 Huber Loss
        L_1 = mean(w_i · HuberLoss(Q_1(s_i,a_i), y_i))
        L_2 = mean(w_i · HuberLoss(Q_2(s_i,a_i), y_i))
        梯度裁剪(max_norm=5.0)后更新 Q_1, Q_2

        # ★ 创新 ⑤：更新 PER 优先级
        δ_i = (|Q_1(s_i,a_i)-y_i| + |Q_2(s_i,a_i)-y_i|) / 2
        p_i = δ_i^α + ε
        D.update_priorities(idx_i, p_i)

        # ---------- Actor 更新（延迟 d=2 步） ----------
        update_time += 1
        if update_time % 2 == 0:
            a_A = π_A(s), a_B = π_B(s)
            # ★ 创新 ①：共识正则化
            L_A = -mean(Q_1(s, a_A)) + 0.01·||a_A - a_B.detach()||²
            L_B = -mean(Q_1(s, a_B)) + 0.01·||a_B - a_A.detach()||²
            梯度裁剪后更新 π_A, π_B

            # 软更新目标网络：θ' ← 0.005·θ + 0.995·θ'
            for each target network: θ' = τ·θ + (1-τ)·θ'

        s ← s', past_action ← a
    end for

    # 每 episode 结束
    OU_noise.decay()     # ★ 创新 ④：σ ← max(σ_min, σ·0.9992)
    β ← min(1.0, β + 0.6/T_max)  # ★ 创新 ⑤：β 线性递增
end for
```

### 2.9 RAV-TD3 核心公式速查

| 编号 | 公式名 | 数学表达式 |
|:---:|-------|-----------|
| ① | 双 Actor 投票 | $a_{voted} = \sum_{k \in \{A,B\}} \frac{\exp(\min_{j} Q_j(s,a_k) / T)}{\sum_m \exp(\min_j Q_j(s,a_m) / T)} \cdot a_k$ |
| ② | 角速度风险增强 | $a[\omega] \leftarrow a[\omega] + risk^2 \cdot (a_{stronger}[\omega] - a[\omega])$ |
| ③ | 共识正则化 | $\mathcal{L} = -\mathbb{E}[Q_1(s, \pi_k(s))] + 0.01 \cdot \|\pi_A(s) - \pi_B(s)\|^2$ |
| ④ | 注意力权重 | $\alpha_i = \text{softmax}(W_a e_i + \frac{1.5}{d_i + 0.05})$ |
| ⑤ | 上下文向量 | $\mathbf{c} = \sum_{i=1}^{36} \alpha_i e_i$ |
| ⑥ | 风险系数 | $risk = \text{clip}(\frac{0.85 - d_{front}}{0.85 - 0.18}, 0, 1)$ |
| ⑦ | 风险感知目标奖励 | $r_{goal} = 22 \cdot \Delta d \cdot (1 - 0.35 \cdot risk)$ |
| ⑧ | OU 噪声 | $x_{t+1} = x_t + 0.20(0 - x_t) + \sigma \cdot \mathcal{N}(0,1)$ |
| ⑨ | PER 优先级 | $p_i = \left(\frac{|\delta_i^{(1)}| + |\delta_i^{(2)}|}{2}\right)^{0.6} + \epsilon$ |
| ⑩ | 重要性采样权重 | $w_i = \left(N \cdot P(i)\right)^{-\beta_t}$ |
| ⑪ | Clipped Double Q | $y = r + \gamma \min_{j=1,2} Q_{\theta_j'}(s', a')$ |
| ⑫ | 目标策略平滑 | $a' = \pi'(s') + \text{clip}(\mathcal{N}(0, \sigma), -c, c)$ |

---

## 3. 系统架构

### 3.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gazebo 仿真环境                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐│
│  │ Limo 无人车│  │ 激光雷达   │  │   IMU     │  │ 机场障碍物模型 ││
│  │ 阿克曼模型 │  │ LaserScan │  │ Odometry  │  │ 机库/飞机等   ││
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └───────────────┘│
└────────┼──────────────┼──────────────┼──────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ROS Topic 通信层                             │
│              /scan    /odom    /cmd_vel                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     强化学习训练节点                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────────┐ │
│  │   Environment   │───▶│    RAV-TD3       │◀───│  ReplayBuffer │ │
│  │   (环境交互)     │    │   (策略学习)    │    │   (经验回放)   │ │
│  └─────────────────┘    └─────────────────┘    └───────────────┘ │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐                      │
│  │ 奖励函数计算     │    │ Actor-Critic网络│                      │
│  │ 状态预处理       │    │  PolarAttention │                      │
│  └─────────────────┘    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
[激光雷达 /scan] ──┐
                  ├──▶ [Environment.getState()] ──▶ [38维状态向量] ──▶ [RAV-TD3.choose_action()]
[里程计 /odom] ────┘                                              │
                                                                  ▼
[目标点位置] ─────────────────────────────────────────────────▶ [PolarLidarAttentionEncoder]
                                                                  │
                                                                  ▼
[cmd_vel] ◀── [动作: (v, ω)] ◀── [Actor-A/B 投票] ◀── [双策略网络]
```

---

## 4. 状态空间与动作空间

### 4.1 状态空间设计

本项目设计的状态空间共 **38 维**，包含激光雷达特征、导航信息和历史动作：

| 状态分量 | 维度 | 描述 | 范围 |
|---------|------|------|------|
| 激光雷达扇区特征 | 36 | 将 360° 激光雷达数据压缩为 36 个扇区，每个扇区取最小距离值 | [0, 6.0] m |
| 航向角 | 1 | 目标点相对机器人朝向的角度 | [-π, π] rad |
| 目标距离 | 1 | 机器人到目标点的欧几里得距离 | [0, ∞) m |

**状态向量结构：**
```python
state = [scan_sector_0, scan_sector_1, ..., scan_sector_35, heading, distance]
#        |<------ 36 维激光雷达特征 ------>||<-- 2 维导航信息 -->|
# 总计: 38 维
```

#### 4.1.1 激光雷达特征降维

原始激光雷达数据（通常为 360 或 720 个数据点）通过扇区划分与最小值池化进行降维：

```python
num_sectors = 36
sector_size = len(scan_range) // num_sectors
downsampled_scan = []

for i in range(num_sectors):
    start_idx = i * sector_size
    end_idx = min(start_idx + sector_size, len(scan_range))
    sector_min = min(scan_range[start_idx:end_idx])
    downsampled_scan.append(sector_min)
```

#### 4.1.2 航向角计算

```python
goal_angle = atan2(goal_y - position.y, goal_x - position.x)
heading = goal_angle - yaw  # 机器人当前偏航角

# 归一化到 [-π, π]
if heading > pi:
    heading -= 2 * pi
elif heading < -pi:
    heading += 2 * pi
```

### 4.2 动作空间设计

动作空间为 **2 维连续向量**，分别控制线速度和角速度：

| 动作分量 | 范围 | 单位 | 物理意义 |
|---------|------|------|---------|
| 线速度 $v$ | [0.0, 0.8] | m/s | 无人车前进速度 |
| 角速度 $\omega$ | [-1.8, 1.8] | rad/s | 无人车转向角速度 |

**动作输出激活函数：**

```python
action[:, 0] = sigmoid(action_raw[:, 0]) * v_max    # 线速度 [0, 0.8]
action[:, 1] = tanh(action_raw[:, 1]) * w_max       # 角速度 [-1.8, 1.8]
```

### 4.3 探索噪声设计

训练时采用**自适应 Ornstein-Uhlenbeck (OU) 噪声过程**进行探索，相比固定高斯噪声具有以下优势：

1. **时间相关性**：相邻时刻的噪声值相关，更适合物理系统的连续控制
2. **均值回归**：噪声值会向均值(0)回归，避免长期偏离
3. **自适应衰减**：随训练进展自动降低噪声强度

```python
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.20, sigma_max=0.35, 
                 sigma_min=0.08, decay_rate=0.9992):
        self.mu = mu              # 均值回归目标
        self.theta = theta        # 均值回归速度
        self.sigma = sigma_max    # 初始噪声强度
        self.sigma_min = sigma_min # 最小噪声强度下限
        self.decay_rate = decay_rate  # 噪声衰减率

    def sample(self):
        """采样OU噪声"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state.copy()

    def decay_noise(self):
        """每episode结束时衰减噪声强度"""
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)
```

实际探索噪声具有**各向异性**：
```python
ou_noise = self.exploration_noise.sample()
noise_scale = [0.50, 1.20]  # 线速度/角速度缩放系数
noise = ou_noise * noise_scale
```

这种设计确保线速度和角速度的探索强度匹配实际物理限制。

---

## 5. 奖励函数设计

奖励函数是强化学习的核心，直接影响智能体学习到的策略质量。本项目设计了**多目标加权奖励函数**，平衡导航效率、避障安全和运动平稳性。

### 5.1 奖励函数组成

| 奖励项 | 公式 | 权重 | 作用 |
|-------|------|------|------|
| 目标推进奖励 | $r_{goal} = 22.0 \times \Delta d \times (1 - 0.35 \times risk)$ | 22.0 | 鼓励向目标移动，风险高时降低权重 |
| 朝向奖励 | $r_{heading} = 1.6 \times \cos(heading)$ | 1.6 | 鼓励朝向目标 |
| 安全间隙奖励 | $r_{clearance} = 2.0 \times \tanh((d_{min} - 0.18) \times 3)$ | 2.0 | 鼓励远离障碍物 |
| 速度奖励 | $r_{speed} = 1.0 \times v \times (1 - 0.5 \times risk)$ | 1.0 | 鼓励前进，风险高时降低 |
| 转向奖励 | $r_{turn} = 1.2 \times |\omega| \times (0.25 + risk)$ | 1.2 | 风险高时鼓励转向避障 |
| 平滑惩罚 | $r_{smooth} = -0.25 \times (|\Delta v| + 0.8 \times |\Delta \omega|)$ | -0.25 | 抑制剧烈动作变化 |
| 风险直行惩罚 | $r_{risky\_straight} = -2.5 \times risk \times \max(0.22 - |\omega|, 0)$ | -2.5 | 高风险时惩罚直行 |
| 风险高速惩罚 | $r_{risky\_speed} = -2.0 \times risk \times \max(v - 0.25, 0)$ | -2.0 | 高风险时惩罚高速 |
| 存活奖励 | $r_{survive} = 0.10$ | 0.1 | 鼓励持续探索 |

### 5.2 风险系数计算

```python
risk = clip(
    (collision_risk_range - front_min_range) / 
    (collision_risk_range - collision_range),
    0.0, 1.0
)
# collision_risk_range = 0.85m, collision_range = 0.18m
```

风险系数在 0~1 之间，越接近障碍物风险越高。

### 5.3 终止条件与惩罚

| 终止条件 | 触发条件 | 奖励值 | 说明 |
|---------|---------|-------|------|
| 碰撞 | $d_{min} < 0.18m$ | -260 | 碰撞障碍物 |
| 到达目标 | $d < 0.35m$ | +420 | 成功到达目标点 |
| 旋转卡死 | 连续 120 步 $v < 0.05$ 且 $\|\omega\| > 0.6$ | -80 | 原地旋转无法前进 |
| 无进度 | 连续 140 步未向目标移动 | -100 | 陷入局部最优 |
| 超时 | 步数达到 1000 | 0 | 回合超时 |

### 5.4 奖励函数实现代码

```python
def setReward(self, state, action, past_action, done):
    heading = state[-2]
    current_distance = state[-1]
    front_min_range = self.latest_front_min

    # 计算进度和风险
    progress = self.past_distance - current_distance
    heading_alignment = math.cos(heading)
    risk = np.clip((0.85 - front_min_range) / (0.85 - 0.18), 0.0, 1.0)

    # 多目标奖励计算
    goal_reward = 22.0 * progress * (1.0 - 0.35 * risk)
    heading_reward = 1.6 * heading_alignment
    clearance_reward = 2.0 * np.tanh((front_min_range - 0.18) * 3.0)
    speed_reward = 1.0 * action[0] * (1.0 - 0.5 * risk)
    turn_reward = 1.2 * abs(action[1]) * (0.25 + risk)
    smooth_penalty = -0.25 * (abs(action[0] - past_action[0]) + 
                               0.8 * abs(action[1] - past_action[1]))
    
    # 风险感知惩罚
    risky_straight_penalty = -2.5 * risk * max(0.22 - abs(action[1]), 0.0)
    risky_speed_penalty = -2.0 * risk * max(action[0] - 0.25, 0.0)
    
    survival_reward = 0.10

    reward = (goal_reward + heading_reward + clearance_reward + 
              speed_reward + turn_reward + smooth_penalty + 
              risky_straight_penalty + risky_speed_penalty + survival_reward)
    
    return reward, done
```

---

## 6. 网络架构

### 6.1 PolarLidarAttentionEncoder（极坐标激光雷达注意力编码器）

这是本项目的核心创新，将激光雷达扇区特征与目标极坐标信息融合：

```
输入: [激光雷达36维 + 目标2维] = 38维

激光雷达分支:
    scan_seq [B, 36, 1]
        ↓
    sector_fc: Linear(1, 48) + LayerNorm + ReLU
        ↓
    sector_embed [B, 36, 48]
        ↓
    注意力计算:
        learned_logits = attn_fc(sector_embed)  [B, 36]
        risk_bias = 1.5 / (scan + 0.05)         [B, 36]
        attn_weights = softmax(learned_logits + risk_bias)
        ↓
    context = bmm(attn_weights, sector_embed) [B, 48]

目标分支:
    goal [B, 2]
        ↓
    goal_mlp: Linear(2, 64) → ReLU → Linear(64, 64) → ReLU
        ↓
    goal_feat [B, 64]

统计特征:
    scan_stats = [min, mean, std] [B, 3]

融合:
    fused = cat([context, goal_feat, scan_stats]) [B, 115]
        ↓
    fusion: Linear(115, 192) + LayerNorm + ReLU
        ↓
    输出: [B, 192]
```

### 6.2 Actor 网络（策略网络）

```
输入层: 38 维状态向量
    ↓
PolarLidarAttentionEncoder: 38 → 192
    ↓
全连接层: 192 → 256 (LayerNorm + ReLU)
    ↓
全连接层: 256 → 2
    ↓
输出层:
    [0] → Sigmoid → × 0.8  → 线速度 v
    [1] → Tanh    → × 1.8  → 角速度 ω
```

**双 Actor 架构：**
- Actor-A 和 Actor-B 两个独立的策略网络
- 通过 Critic 评估进行投票决策
- 共识正则化促进策略一致性

### 6.3 Critic 网络（价值网络）

```
输入层: 38 维状态 + 2 维动作 = 40 维向量
    ↓
状态编码:
    PolarLidarAttentionEncoder: 38 → 192
        ↓
动作编码:
    Linear(2, 64) + ReLU
        ↓
拼接: [状态特征 192, 动作特征 64] = 256
    ↓
全连接层: 256 → 256 (LayerNorm + ReLU)
    ↓
全连接层: 256 → 1
    ↓
输出层: Q 值标量
```

**双重 Critic：**
- Critic-1 和 Critic-2 两个独立的价值网络
- 目标值取两者较小值：$y = r + \gamma \min(Q_1', Q_2')$

### 6.4 网络参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Actor 学习率 α | 0.0001 | Adam 优化器 |
| Critic 学习率 β | 0.0001 | Adam 优化器 |
| 扇区嵌入维度 | 48 | 每个激光雷达扇区的嵌入维度 |
| 融合特征维度 | 192 | 编码器输出维度 |
| Actor FC1 | 512 | 第一层全连接（实际使用编码器输出192） |
| Actor FC2 | 256 | 第二层全连接 |
| Critic FC1 | 512 | 第一层全连接 |
| Critic FC2 | 256 | 第二层全连接 |
| 梯度裁剪 | 5.0 | 防止梯度爆炸 |

---

## 7. 项目文件结构

```
DRL_ws/
├── src/
│   ├── limoRL/                    # 强化学习核心包
│   │   ├── launch/
│   │   │   ├── limo_TD3_train.launch    # 训练启动文件
│   │   │   └── limo_TD3_test.launch     # 测试启动文件
│   │   ├── scripts/
│   │   │   ├── TD3/
│   │   │   │   ├── TD3Net.py      # RAV-TD3 神经网络实现
│   │   │   │   ├── Environment.py # 强化学习环境
│   │   │   │   ├── buffer.py      # 经验回放缓冲区
│   │   │   │   ├── per_buffer.py  # 优先经验回放缓冲区(PER)
│   │   │   │   └── respawnGoal.py # 目标点重置
│   │   │   ├── limo_td3.py        # 训练主程序
│   │   │   ├── limo_td3_test.py   # 测试主程序
│   │   │   └── utils.py           # 可视化工具
│   │   ├── train/
│   │   │   └── TD3/
│   │   │       └── model/         # 保存的模型文件
│   │   ├── models/                # Gazebo 模型文件
│   │   └── CMakeLists.txt
│   │
│   ├── limo_gazebo_sim/           # Gazebo 仿真包
│   │   ├── launch/
│   │   │   └── limoEnv.launch     # 仿真环境启动
│   │   ├── worlds/
│   │   │   ├── limo_state.world   # 机场仿真世界
│   │   │   └── limo_static.world  # 静态障碍世界
│   │   └── config/
│   │       └── limo_ackerman_control.yaml
│   │
│   ├── limo_description/          # 机器人模型包
│   │   └── urdf/
│   │       └── limo_ackerman.xacro
│   │
│   ├── limo_bringup/              # 导航启动包
│   │   └── launch/
│   │       └── limo_navigation_diff.launch
│   │
│   └── vehicle_simulator/         # 车辆仿真器
│
├── build/                         # 编译输出
├── devel/                         # 开发环境
└── README.md
```

### 7.1 核心文件说明

| 文件路径 | 功能描述 |
|---------|---------|
| `limoRL/scripts/TD3/TD3Net.py` | RAV-TD3 算法核心实现，包含双 Actor/Critic 网络、双策略投票机制、自适应OU噪声 |
| `limoRL/scripts/TD3/Environment.py` | 强化学习环境，处理传感器数据、计算奖励、执行动作 |
| `limoRL/scripts/TD3/buffer.py` | 标准经验回放缓冲区，存储和采样训练数据 |
| `limoRL/scripts/TD3/per_buffer.py` | 优先经验回放(PER)缓冲区，基于SumTree实现优先级采样 |
| `limoRL/scripts/TD3/respawnGoal.py` | 目标点管理，支持候选点和随机采样 |
| `limoRL/scripts/limo_td3.py` | 训练脚本主程序 |
| `limoRL/scripts/limo_td3_test.py` | 测试脚本主程序 |

---

## 8. 环境配置与依赖

### 8.1 系统要求

| 组件 | 版本要求 |
|------|---------|
| 操作系统 | Ubuntu 20.04 LTS |
| ROS | Noetic Ninjemys |
| Python | 3.8+ |
| CUDA | 11.0+ (可选，用于GPU加速) |

### 8.2 ROS 依赖安装

```bash
# 安装 ROS Noetic 基础依赖
sudo apt update
sudo apt install -y ros-noetic-desktop-full

# 安装 Gazebo 和导航相关包
sudo apt install -y \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-navigation \
    ros-noetic-slam-gmapping \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-controller-manager \
    ros-noetic-velocity-controllers \
    ros-noetic-position-controllers

# 安装 Python 依赖
pip3 install torch torchvision numpy matplotlib
```

### 8.3 工作空间编译

```bash
cd ~/DRL_ws
catkin_make
source devel/setup.bash

# 添加到 .bashrc
echo "source ~/DRL_ws/devel/setup.bash" >> ~/.bashrc
```

---

## 9. 运行步骤

### 9.1 训练流程

#### 步骤 1：启动 Gazebo 仿真环境

```bash
# 终端 1：启动 Gazebo 仿真
roslaunch limo_gazebo_sim limoEnv.launch world_name:=$(find limo_gazebo_sim)/worlds/limo_state.world
```

#### 步骤 2：启动 RAV-TD3 训练节点

```bash
# 终端 2：启动训练
roslaunch limoRL limo_TD3_train.launch
```

**训练参数说明：**

| 参数名 | 默认值 | 说明 |
|-------|--------|------|
| `random_goal_on_reset` | true | 每回合随机重置目标点 |
| `collision_range` | 0.18 | 碰撞检测距离(m) |
| `goal_tolerance` | 0.35 | 到达目标容差(m) |
| `random_action_steps` | 1500 | 纯随机探索步数 |
| `max_steps` | 1000 | 每回合最大步数 |
| `action_noise` | 0.12 | 动作探索噪声（已迁移至自适应OU噪声） |
| `save_interval` | 20 | 模型保存间隔(回合) |
| `use_per` | true | 是否启用优先经验回放(PER) |
| `per_alpha` | 0.6 | PER优先级指数(0=均匀采样, 1=完全优先采样) |
| `pretrained_ckpt_dir` | "" | 预训练模型目录(课程学习，默认空) |
| `pretrained_episode` | 0 | 预训练模型轮次(课程学习，默认0) |

### 9.2 测试流程

#### 步骤 1：启动仿真环境

```bash
roslaunch limo_gazebo_sim limoEnv.launch
```

#### 步骤 2：启动测试节点

```bash
# 测试最新模型
roslaunch limoRL limo_TD3_test.launch

# 测试指定回合模型
roslaunch limoRL limo_TD3_test.launch episode:=500

# 指定测试回合数
roslaunch limoRL limo_TD3_test.launch test_episodes:=50
```

### 9.3 可视化调试

```bash
# 启动 RViz 查看传感器数据
roslaunch limo_bringup limo_navigation_diff.launch

# 查看 ROS 话题
rostopic list
rostopic echo /scan          # 激光雷达数据
rostopic echo /odom          # 里程计数据
rostopic echo /cmd_vel       # 控制指令
```

### 9.4 课程学习/微调模式

课程学习通过**加载预训练权重**实现多阶段训练，适用于以下场景：
- **分阶段训练**：第一阶段在简单场景训练，第二阶段加载权重在复杂场景微调
- **迁移学习**：加载相似任务的预训练模型进行微调
- **恢复训练**：从中断的检查点继续训练

#### 9.4.1 第一阶段：基础训练

在基础环境中训练，保存模型权重：
```bash
# 第一阶段训练完成后，模型保存在默认目录
# /home/zjs/DRL_ws/src/limoRL/train/TD3/model/
```

#### 9.4.2 第二阶段：加载预训练权重微调

```bash
# 指定预训练模型目录和轮次，启动微调训练
roslaunch limoRL limo_TD3_train.launch \
    pretrained_ckpt_dir:="/home/zjs/DRL_ws/src/limoRL/train/TD3/model" \
    pretrained_episode:=6120
```

**微调机制说明：**
1. 只加载主网络参数（actor_a, actor_b, critic1, critic2）
2. 目标网络通过软更新同步（tau=1.0 硬拷贝）
3. 探索噪声自动重置，从较高强度开始衰减
4. 训练从指定episode继续，奖励历史记录重新开始

### 9.5 模型文件说明

训练过程中保存的模型文件位于 `src/limoRL/train/TD3/model/` 目录：

| 文件名 | 说明 |
|-------|------|
| `TD3_actor_{episode}.pth` | Actor-A 策略网络权重 |
| `TD3_actor_b_{episode}.pth` | Actor-B 策略网络权重 |
| `TD3_target_actor_{episode}.pth` | Target Actor-A 网络权重 |
| `TD3_target_actor_b_{episode}.pth` | Target Actor-B 网络权重 |
| `TD3_critic1_{episode}.pth` | Critic-1 网络权重 |
| `TD3_critic2_{episode}.pth` | Critic-2 网络权重 |
| `TD3_target_critic1_{episode}.pth` | Target Critic-1 网络权重 |
| `TD3_target_critic2_{episode}.pth` | Target Critic-2 网络权重 |

---

## 10. 关键代码解读

### 10.1 双策略投票机制

```python
def _vote_actions(self, state_tensor, use_target=False):
    # 获取两个 Actor 的动作
    if use_target:
        action_a = self.target_actor_a.forward(state_tensor)
        action_b = self.target_actor_b.forward(state_tensor)
    else:
        action_a = self.actor_a.forward(state_tensor)
        action_b = self.actor_b.forward(state_tensor)

    # Critic 评估两个动作
    with T.no_grad():
        q_a = T.min(
            self.critic1.forward(state_tensor, action_a),
            self.critic2.forward(state_tensor, action_a),
        )
        q_b = T.min(
            self.critic1.forward(state_tensor, action_b),
            self.critic2.forward(state_tensor, action_b),
        )

    # Softmax 投票
    score = T.cat([q_a, q_b], dim=1) / self.vote_temp
    vote_weight = F.softmax(score, dim=1)
    voted_action = vote_weight[:, 0:1] * action_a + vote_weight[:, 1:2] * action_b

    # 风险感知角速度增强
    scan = state_tensor[:, :self.scan_dim]
    front_min = T.min(scan, dim=1, keepdim=True).values
    risk = T.clamp(
        (self.vote_risk_range - front_min) / (self.vote_risk_range - self.vote_collision_range),
        0.0, 1.0,
    )
    # 选择转向更强的动作
    stronger_turn = T.where(T.abs(action_a[:, 1:2]) >= T.abs(action_b[:, 1:2]), 
                            action_a[:, 1:2], action_b[:, 1:2])
    min_turn = self.angular_vote_boost * risk
    voted_w = voted_action[:, 1:2]
    voted_w = T.where(T.abs(voted_w) < min_turn, 
                      T.sign(stronger_turn + 1e-6) * min_turn, voted_w)
    
    voted_action[:, 1:2] = T.clamp(voted_w, -self.action_limit_w, self.action_limit_w)
    return voted_action, action_a, action_b, vote_weight
```

### 10.2 注意力编码器

```python
class PolarLidarAttentionEncoder(nn.Module):
    def forward(self, state):
        scan = state[:, :self.scan_dim]      # [B, 36]
        goal = state[:, self.scan_dim:self.scan_dim + self.goal_dim]  # [B, 2]

        # 扇区嵌入 [B, 36, 1] -> [B, 36, 48]
        scan_seq = scan.unsqueeze(-1)
        sector_embed = self.sector_fc(scan_seq)
        sector_embed = self.sector_ln(sector_embed)
        sector_embed = F.relu(sector_embed)

        # 可学习注意力 + 显式风险偏置（逆距离）
        learned_logits = self.attn_fc(sector_embed).squeeze(-1)  # [B, 36]
        risk_bias = self.risk_bias_gain / (scan + 0.05)          # [B, 36]
        attn_logits = learned_logits + risk_bias
        attn_weights = F.softmax(attn_logits, dim=-1)

        # 加权求和得到上下文向量
        context = T.bmm(attn_weights.unsqueeze(1), sector_embed).squeeze(1)  # [B, 48]

        # 统计特征
        scan_min = T.min(scan, dim=1, keepdim=True).values
        scan_mean = T.mean(scan, dim=1, keepdim=True)
        scan_std = T.std(scan, dim=1, keepdim=True, unbiased=False)
        scan_stats = T.cat([scan_min, scan_mean, scan_std], dim=1)

        # 目标特征
        goal_feat = self.goal_mlp(goal)

        # 融合
        fused = T.cat([context, goal_feat, scan_stats], dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_ln(fused)
        fused = F.relu(fused)
        return fused, attn_weights
```

### 10.3 Critic 更新（支持PER）

```python
def learn(self):
    if not self.memory.ready():
        return

    # 支持优先经验回放(PER)
    if self.use_per:
        states, actions, rewards, states_, terminals, indices, weights = self.memory.sample_buffer()
        weights_tensor = T.as_tensor(weights, device=device)
    else:
        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        weights_tensor = T.ones_like(T.as_tensor(rewards, device=device))
    
    with T.no_grad():
        # 目标动作添加噪声（目标策略平滑）
        next_actions_tensor, _, _, _ = self._vote_actions(next_states_tensor, use_target=True)
        action_noise = T.tensor(
            np.stack([
                np.random.normal(0.0, self.policy_noise * 0.6, size=batch_size),
                np.random.normal(0.0, self.policy_noise * 1.4, size=batch_size),
            ], axis=1),
            dtype=T.float,
        ).to(device)
        action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
        next_actions_tensor = next_actions_tensor + action_noise
        
        # 裁剪目标动作
        next_actions_tensor[:, 0] = T.clamp(next_actions_tensor[:, 0], 0.0, self.action_limit_v)
        next_actions_tensor[:, 1] = T.clamp(next_actions_tensor[:, 1], -self.action_limit_w, self.action_limit_w)
        
        # 双重 Q 值计算，取较小值
        q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
        q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
        q1_[terminals_tensor] = 0.0
        q2_[terminals_tensor] = 0.0
        critic_val = T.min(q1_, q2_)  # 关键：取较小值防止过估计
        target = rewards_tensor + self.gamma * critic_val

    # 当前 Q 值
    q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
    q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

    # Huber Loss（PER模式下使用重要性采样权重）
    if self.use_per:
        critic1_loss = T.mean(weights_tensor * F.huber_loss(q1, target.detach(), reduction='none'))
        critic2_loss = T.mean(weights_tensor * F.huber_loss(q2, target.detach(), reduction='none'))
    else:
        critic1_loss = F.huber_loss(q1, target.detach())
        critic2_loss = F.huber_loss(q2, target.detach())

    # 反向传播
    self.critic1.optimizer.zero_grad()
    critic1_loss.backward()
    T.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=5.0)
    self.critic1.optimizer.step()
    # ... Critic-2 同理
    
    # PER模式下更新经验优先级
    if self.use_per:
        td_errors = T.abs(q1 - target) + T.abs(q2 - target)
        self.memory.update_priorities(indices, td_errors.cpu().numpy())
```

### 10.4 Actor 更新（带共识正则化）

```python
# 延迟策略更新
self.update_time += 1
if self.update_time % self.delay_time != 0:
    return

# 计算两个 Actor 的损失
act_a = self.actor_a.forward(states_tensor)
act_b = self.actor_b.forward(states_tensor)

qa = T.min(self.critic1.forward(states_tensor, act_a), 
           self.critic2.forward(states_tensor, act_a))
qb = T.min(self.critic1.forward(states_tensor, act_b), 
           self.critic2.forward(states_tensor, act_b))

# 共识正则化：促进两个策略趋于一致
vote_consensus_a = T.mean((act_a - act_b.detach()) ** 2)
vote_consensus_b = T.mean((act_b - act_a.detach()) ** 2)

actor_a_loss = -T.mean(qa) + 0.01 * vote_consensus_a
actor_b_loss = -T.mean(qb) + 0.01 * vote_consensus_b

# 反向传播更新
self.actor_a.optimizer.zero_grad()
actor_a_loss.backward()
T.nn.utils.clip_grad_norm_(self.actor_a.parameters(), max_norm=5.0)
self.actor_a.optimizer.step()
# ... Actor-B 同理

# 软更新目标网络
self.update_network_parameters()
```

---

## 11. 实验结果与分析

### 11.1 训练曲线特征

典型的 RAV-TD3 训练曲线呈现以下阶段：

| 阶段 | 回合范围 | 特征描述 |
|------|---------|---------|
| **初期探索** | 0-100 | 奖励波动大，智能体主要进行随机探索，成功率 < 20% |
| **快速学习** | 100-300 | 奖励快速上升，智能体学会基础避障和朝向目标 |
| **策略收敛** | 300-500 | 奖励趋于稳定，成功率达到 70-80% |
| **精细优化** | 500+ | 奖励小幅波动，策略持续优化，成功率 > 85% |

### 11.2 性能指标

| 指标 | 数值 | 说明 |
|-----|------|------|
| 导航成功率 | > 85% | 在测试场景中成功到达目标的比例 |
| 平均回合奖励 | 400-600 | 成功回合的典型奖励范围 |
| 避障响应时间 | < 100ms | 从检测到障碍物到策略输出的延迟 |
| 碰撞率 | < 10% | 在测试场景中的碰撞次数占比 |
| 平均步数 | 200-600 | 成功到达目标所需的平均步数 |

### 11.3 消融实验对比

| 配置 | 成功率 | 说明 |
|-----|-------|------|
| 单 Actor + 单 Critic (DDPG) | 50% | 基线算法，存在过估计问题 |
| 单 Actor + 双 Critic (TD3基础) | 50% | 解决过估计，但策略单一 |
| 双 Actor + 注意力编码器 + 风险感知奖励 | 85%+ | 双策略投票，决策更稳健 |
| **RAV-TD3（完整算法）** | 92% | + 自适应OU噪声 + 优先经验回放 |

### 11.4 训练曲线分析

#### 阶段一：基础TD3

![经典TD3](https://i-blog.csdnimg.cn/direct/0151568d3896403289d2359681648308.png)

#### 阶段二：RAV-TD3 v1（双Actor投票 + 风险感知编码器）

![RAV-TD3 v1](https://i-blog.csdnimg.cn/direct/fd605c5c61f941a08056afd24910e9e8.png)

**RAV-TD3 v1 训练问题分析：**

| 问题 | 现象 | 可能原因 |
|------|------|---------|
| **方差过大** | 成功episode奖励从600到1200不等，标准差估计在150-200 | 目标点难度差异大；策略泛化能力不足 |
| **持续负奖励** | 训练后期仍有10-15%的episode奖励为负（~-250到-50） | 策略在某些场景下碰撞/卡死，鲁棒性不够 |
| **异常高奖励** | 出现>3000甚至5826的极端值 | 奖励函数设计有漏洞，agent找到了"刷分"策略；某些目标点特别容易，agent在附近绕圈累积奖励；超时未惩罚反而给了持续正向奖励 |

#### 阶段三：RAV-TD3 v2（引入自适应OU噪声+PER+课程学习）
以下是根据您提供的四点改进整理成的 Markdown 格式对比表格：

| 改进项 | 原代码缺陷 | 新代码改进逻辑 | 核心变化总结 |
| :--- | :--- | :--- | :--- |
| **1. 最小转角强制机制** | `voted_w` 在风险存在时被直接**覆盖**为最小转向值 (`T.where` 强制替换)，导致即便无障可直行也因最小转速被强制转向。 | 改用**加权混合策略**：<br>`voted_w = voted_w + risk_weight * (enhanced_w - voted_w)`<br>其中 `risk_weight = risk ** 2`，实现平滑过渡。 | 由 **强制替换** 改为 **平滑加权**，低风险时保留网络直行意图，高风险时按需增强避障。 |
| **2. 转向方向隐式偏置** | `sign(stronger_turn + 1e-6)` 当 `stronger_turn` 接近零时，由于添加了 `1e-6` 偏置，导致转向方向**永远偏向正方向**。 | 方向判断改用原始投票方向：<br>`sign(voted_w + 1e-6)`。 | 尊重网络**原本的转向意图**，消除人为引入的单侧转向偏置风险。 |
| **3. Risk 敏感区域计算** | 使用**全局最小激光值** `T.min(scan)` 计算风险，侧后方近距离障碍物也会误触发高 `risk` 导致异常避障。 | 仅提取**前方特定扇区**计算风险：<br>`front_scan = scan[:, center - width : center + width + 1]`<br>仅覆盖前方约 1/3 范围。 | 风险判断仅聚焦**前方有效区域**，避免侧后方障碍物对避障策略的干扰。 |
| **4. 与奖励函数同向放大** | 系数较高 (`0.35`) 且使用强制覆盖，与训练奖励中的转向鼓励项**同向线性叠加**，易导致转向过度或震荡。 | 1. 降低系数为 `0.25`。<br>2. 引入 `tanh(risk * 3.0)` 饱和机制。<br>3. 采用 `risk_weight = risk ** 2` 抑制低风险增强。 | 降低增益幅度并使用**饱和函数**抑制高风险区线性增长，避免与奖励函数**重复激励**造成过拟合。 |

![RAV-TD3 v2](https://i-blog.csdnimg.cn/direct/a4a301b2f78f4dc6b219d8ae298bc6ef.png)
![RAV-TD3 v2_2](https://i-blog.csdnimg.cn/direct/4206e2137ccb4b679eb5772e75c38995.png)
可达95%-100%成功率



---

## 12. 总结与展望

### 12.1 项目总结

本项目成功实现了基于深度强化学习的无人车自主导航与动态避障系统，主要贡献包括：

1. **算法创新**：
   - 实现了 RAV-TD3 (Risk-Aware Voting TD3) 算法，引入双策略投票机制
   - 设计了风险感知注意力编码器，提升障碍物感知能力
   - 提出了多目标加权奖励函数，平衡效率与安全

2. **工程实践**：
   - 构建了完整的 ROS-Gazebo-PyTorch 训练测试 pipeline
   - 实现了动态目标点生成，避免策略退化
   - 设计了丰富的奖励塑形机制，加速收敛

3. **性能表现**：
   - 在复杂机场仿真环境中达到 95%+ 的导航成功率
   - 实现了端到端的传感器到控制指令的直接映射
   - 具备良好的泛化能力和避障安全性

### 12.2 改进方向

| 方向 | 方法 | 预期效果 |
|-----|------|---------|
| **算法优化** | 引入 SAC (Soft Actor-Critic) | 进一步提升样本效率 |
| **安全保证** | 添加安全约束层 (Safety Layer) | 确保近距避障绝对安全 |
| **迁移学习** | 从仿真到真实环境的域随机化 | 实现 Sim2Real 迁移 |
| **多智能体** | 多车协同导航 | 支持机场多车调度场景 |
| **动态避障** | 引入行人/车辆运动预测 | 应对动态障碍物场景 |

### 12.3 应用前景

本项目的研究成果可应用于：

- **机场无人转运车**：行李、货物自动运输
- **工业物流机器人**：仓库、工厂内部搬运
- **智能仓储系统**：货架间自主导航
- **园区自动驾驶**：封闭/半封闭区域接驳服务
- **服务机器人**：室内导航与避障

---

## 13. 参考文献

1. Fujimoto S, van Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]. International Conference on Machine Learning (ICML), 2018.
2. Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.
3. Silver D, Lever G, Heess N, et al. Deterministic policy gradient algorithms[C]. International Conference on Machine Learning (ICML), 2014.
4. Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]. ICML, 2018.

---

## 14. 模型与地图说明

> **⚠️ 重要说明：开源版本与演示视频的地图差异**

本项目的演示视频中使用了基于特定保密模型（如战斗机等）构建的高保真仿真地图。由于这些模型文件涉及保密内容，不便在开源仓库中分享，因此**开源版本中的 Gazebo 仿真地图已替换为通用的障碍物模型**（立方体、圆柱体、屏障等简单几何体）。

**但核心强化学习算法 RAV-TD3 (Risk-Aware Voting TD3) 完全未作任何改动**。您可以将 RAV-TD3 直接部署到自己的 Gazebo 地图及模型环境中，不影响训练效果与部署效果。算法仅依赖 LiDAR 激光数据和目标位姿信息进行决策，与地图视觉外观无关。

如您需要在自己的场景中使用 RAV-TD3，只需：
1. 创建或使用自己的 Gazebo 世界文件（.world）
2. 在启动文件中指定 `world_name` 参数指向您的世界文件
3. 确保 LiDAR 话题（`/scan`）和里程计话题（`/odom`）的格式与 ROS 标准一致

算法会自动适应不同的障碍物布局与场景结构。

---

## 15. 附录：快速开始命令速查

```bash
# 1. 编译工作空间
cd ~/DRL_ws && catkin_make && source devel/setup.bash

# 2. 启动仿真环境（终端1）
source devel/setup.bash 
roslaunch limo_gazebo_sim limoEnv.launch
```

### 标准训练模式（默认开启PER，课程学习关闭）

```bash
# 3. 启动训练（终端2）
roslaunch limoRL limo_TD3_train.launch

# 关闭PER训练（使用标准经验回放）
roslaunch limoRL limo_TD3_train.launch use_per:=false

# 调整PER优先级指数
roslaunch limoRL limo_TD3_train.launch per_alpha:=0.7
```

### 课程学习/微调模式

```bash
# 加载预训练权重进行微调（两阶段训练）
# 第一阶段：在简单场景训练并保存模型
# 第二阶段：加载第一阶段模型，在复杂场景微调
roslaunch limoRL limo_TD3_train.launch \
    pretrained_ckpt_dir:="/home/zjs/DRL_ws/src/limoRL/train/TD3/model" \
    pretrained_episode:=6120

# 课程学习 + 关闭PER
roslaunch limoRL limo_TD3_train.launch \
    pretrained_ckpt_dir:="/home/zjs/DRL_ws/src/limoRL/train/TD3/model" \
    pretrained_episode:=6120 \
    use_per:=false
```

### 测试与可视化

```bash
# 4. 启动测试（终端2，先停止训练）
roslaunch limoRL limo_TD3_test.launch episode:=500

# 5. 查看话题
rostopic list
rostopic hz /scan

# 6. 可视化
roslaunch limo_bringup limo_navigation_diff.launch
```

---
**项目维护者**：基于开源算法TD3改进，实现RAV-TD3 (Risk-Aware Voting TD3)

**最后更新**：2026-05-25

---

## 16. 开源协议

### 16.1 GNU General Public License v3.0

```
RAV-TD3: 基于风险感知投票TD3的无人车自主导航系统
Copyright (C) 2024-2026

本程序是自由软件：您可以依据自由软件基金会发布的
GNU通用公共许可证（第三版或其任何更新版本）的条款，
重新分发和/或修改它。

本程序的分发是希望它能有用，
但没有任何担保；甚至没有适销性或特定用途适用性的隐含担保。
详情请参见GNU通用公共许可证。

您应该已随本程序收到GNU通用公共许可证的副本。
如果没有，请参阅 <https://www.gnu.org/licenses/>。
```

**GPLv3 协议要点**：

| 权利 | 说明 |
|------|------|
| **自由使用** | 您可以出于任何目的运行本程序 |
| **自由研究** | 您可以研究程序如何工作，并按需修改 |
| **自由分发** | 您可以重新分发副本 |
| **自由改进** | 您可以分发修改后的版本 |
| **Copyleft（相同方式共享）** | 任何衍生作品也必须以GPLv3协议发布 |
| **专利保护** | 明确的专利授权，保护用户免受专利诉讼 |
| **反Tivo化** | 防止硬件制造商限制用户运行修改后的软件 |

完整的许可文本请参见：[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)。

---

## 17. 致谢

本项目基于开源的TD3算法进行扩展，实现了RAV-TD3 (Risk-Aware Voting TD3) 强化学习框架。感谢ROS、Gazebo和PyTorch开源社区提供的优秀工具和基础设施。

---

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

*本项目遵循GNU通用公共许可证v3.0 — 详见[开源协议](#16-开源协议)章节。*

