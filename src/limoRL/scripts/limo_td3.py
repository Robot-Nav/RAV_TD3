#!/usr/bin/env python3
"""
================================================================================
RAV-TD3 训练主程序 (limo_td3.py)
================================================================================
功能说明：
    启动强化学习环境并执行 RAV-TD3 (Risk-Aware Voting TD3) 在线训练循环。
    支持自适应OU噪声、优先经验回放(PER)、课程学习、损失记录等特性。

训练流程：
    1. 初始化环境和智能体
    2. 随机探索阶段（前 N 步随机动作填充缓冲区）
    3. RAV-TD3 策略训练阶段（双Actor投票 + 风险感知编码）
    4. 定期保存模型、绘制奖励曲线、记录训练损失

增强功能：
    - 分文件夹保存模型（actor/actor_b/critic/target）
    - 训练损失记录（critic_loss, actor_loss）
    - 每60回合保存模型和奖励曲线
    - 异常检测与自动恢复
"""

import numpy as np
import math
import time
import rospy
from std_msgs.msg import Float32MultiArray
import os
import sys
import copy
import json

# 确保当前脚本目录在 Python 路径中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from TD3.TD3Net import TD3
from TD3.Environment import Env
from utils import plotLearning


if __name__=='__main__':
    # ============================================================================
    # 常量定义与ROS参数读取
    # ============================================================================
    PI = math.pi
    ACTION_DIMENSION = 2                          # 动作空间维度：[线速度, 角速度]
    
    # 动作限制参数
    ACTION_V_MAX = float(rospy.get_param('~action_v_max', 0.6))     # 线速度上限 (m/s)
    ACTION_W_MAX = float(rospy.get_param('~action_w_max', 1.0))     # 角速度上限 (rad/s)
    RANDOM_V_MIN = float(rospy.get_param('~random_v_min', 0.10))    # 随机探索最小线速度
    RANDOM_W_MAX = float(rospy.get_param('~random_w_max', 0.9))     # 随机探索最大角速度
    
    # 噪声参数
    ACTION_NOISE = float(rospy.get_param('~action_noise', 0.1))     # 训练探索噪声标准差
    POLICY_NOISE = float(rospy.get_param('~policy_noise', 0.2))     # 策略噪声标准差（目标策略平滑）
    POLICY_NOISE_CLIP = float(rospy.get_param('~policy_noise_clip', 0.5))  # 策略噪声裁剪范围
    
    # 训练控制参数
    MAX_STEPS = int(rospy.get_param('~max_steps', 800))             # 每 episode 最大步数（防止无限绕圈）
    MAX_EPISODES = int(rospy.get_param('~max_episodes', 5000))      # 训练总 episode 数上限
    REWARD_FAIL_LIMIT = float(rospy.get_param('~reward_fail_limit', -500.0))  # 奖励失败阈值
    RANDOM_ACTION_STEPS = int(rospy.get_param('~random_action_steps', 5000))  # 随机探索总步数
    REPLAY_BUFFER_SIZE = int(rospy.get_param('~replay_buffer_size', 200000))  # 经验回放缓冲区大小
    REPLAY_BATCH_SIZE = int(rospy.get_param('~replay_batch_size', 128))       # 训练批量大小
    CUDA_EMPTY_CACHE_INTERVAL = int(rospy.get_param('~cuda_empty_cache_interval', 400))  # CUDA 显存清理间隔
    
    # 网络结构参数
    ACTOR_FC1_DIM = int(rospy.get_param('~actor_fc1_dim', 512))     # Actor 第一层全连接维度
    ACTOR_FC2_DIM = int(rospy.get_param('~actor_fc2_dim', 256))     # Actor 第二层全连接维度
    CRITIC_FC1_DIM = int(rospy.get_param('~critic_fc1_dim', 512))   # Critic 第一层全连接维度
    CRITIC_FC2_DIM = int(rospy.get_param('~critic_fc2_dim', 256))   # Critic 第二层全连接维度
    
    # 保存与日志参数（修改为每60次保存）
    SAVE_INTERVAL = max(1, int(rospy.get_param('~save_interval', 60)))          # 模型保存间隔（默认60回合）
    PLOT_INTERVAL = max(1, int(rospy.get_param('~plot_interval', 60)))          # 奖励曲线绘制间隔（默认60回合）
    STEP_LOG = bool(rospy.get_param('~step_log', False))            # 是否记录每步详情
    STEP_LOG_INTERVAL = max(1, int(rospy.get_param('~step_log_interval', 20)))  # 步日志间隔
    
    # 路径配置
    CKPT_DIR = "/home/zjs/DRL_ws/src/limoRL/train/TD3/model"        # 模型保存根路径
    SAVE_FIGURE_PATH = "/home/zjs/DRL_ws/src/limoRL/train/TD3/png/"  # 奖励曲线保存路径
    LOSS_LOG_PATH = "/home/zjs/DRL_ws/src/limoRL/train/TD3/loss/"   # 损失记录保存路径
    
    # PER与课程学习参数
    USE_PER = bool(rospy.get_param('~use_per', True))               # 是否使用优先经验回放
    PER_ALPHA = float(rospy.get_param('~per_alpha', 0.6))           # PER 优先级指数
    PRETRAINED_CKPT_DIR = rospy.get_param('~pretrained_ckpt_dir', '')  # 预训练模型目录（课程学习）
    PRETRAINED_EPISODE = int(rospy.get_param('~pretrained_episode', 0))  # 预训练模型轮次
    BEST_SCORE_WINDOW = max(1, int(rospy.get_param('~best_score_window', 50)))
    MIN_EPISODES_FOR_BEST = max(1, int(rospy.get_param('~min_episodes_for_best', 100)))
    EARLY_STOP_PATIENCE = max(0, int(rospy.get_param('~early_stop_patience', 0)))
    BEST_IMPROVE_DELTA = float(rospy.get_param('~best_improve_delta', 1.0))

    # ============================================================================
    # ROS节点初始化
    # ============================================================================
    rospy.init_node('limo_td3')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)      # 结果发布器
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)  # 动作发布器
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time = time.time()
    
    # ============================================================================
    # 创建环境与智能体
    # ============================================================================
    env = Env(action_dim=ACTION_DIMENSION)
    init_state = env.reset()  # 首次 reset 获取状态维度
    state_dim = int(init_state.shape[0])
    rospy.loginfo("Detected RAV-TD3 state_dim: %d", state_dim)

    # 初始化RAV-TD3智能体
    agent = TD3(alpha=0.0001, beta=0.0001, state_dim=state_dim, action_dim=2, 
                actor_fc1_dim=ACTOR_FC1_DIM, actor_fc2_dim=ACTOR_FC2_DIM,
                action_limit_v=ACTION_V_MAX, action_limit_w=ACTION_W_MAX, 
                critic_fc1_dim=CRITIC_FC1_DIM, critic_fc2_dim=CRITIC_FC2_DIM,
                ckpt_dir=CKPT_DIR, gamma=0.99, tau=0.005, 
                action_noise=ACTION_NOISE, policy_noise=POLICY_NOISE,
                policy_noise_clip=POLICY_NOISE_CLIP,
                delay_time=2, max_size=REPLAY_BUFFER_SIZE,
                batch_size=REPLAY_BATCH_SIZE,
                cuda_empty_cache_interval=CUDA_EMPTY_CACHE_INTERVAL,
                use_per=USE_PER, per_alpha=PER_ALPHA)

    # ============================================================================
    # 训练状态初始化
    # ============================================================================
    past_action = np.zeros(ACTION_DIMENSION)     # 历史动作初始化
    
    # 创建保存目录
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAVE_FIGURE_PATH, exist_ok=True)
    os.makedirs(LOSS_LOG_PATH, exist_ok=True)
    
    # 历史记录文件
    score_history_file = os.path.join(SAVE_FIGURE_PATH, 'score_history.txt')
    loss_history_file = os.path.join(LOSS_LOG_PATH, 'loss_history.txt')
    best_model_meta_file = os.path.join(CKPT_DIR, 'best_model.json')
    
    # 训练历史数据
    score_history = []                           # 奖励历史记录
    loss_history = []                            # 损失历史记录
    best_window_score = -np.inf
    best_episode = -1
    no_improve_episodes = 0

    # ============================================================================
    # 课程学习：加载预训练权重
    # ============================================================================
    if PRETRAINED_CKPT_DIR and PRETRAINED_EPISODE > 0:
        rospy.loginfo("=== 课程学习/微调模式：加载预训练权重 ===")
        rospy.loginfo("Pretrained model dir: %s, episode: %d", PRETRAINED_CKPT_DIR, PRETRAINED_EPISODE)
        agent.load_pretrained_weights(PRETRAINED_CKPT_DIR, PRETRAINED_EPISODE)
        agent.start_epoch = PRETRAINED_EPISODE
        rospy.loginfo("预训练权重加载完成，从 episode %d 继续训练", agent.start_epoch)

    # ============================================================================
    # 训练主循环
    # ============================================================================
    total_steps = 0
    episode_losses = []  # 记录每个episode的平均损失
    
    for e in range(agent.start_epoch+1, MAX_EPISODES):
        # 每回合重置探索噪声（OU噪声）
        agent.reset_exploration_noise()
        
        # 首次 episode 复用 init_state，后续 episodes 调用 env.reset()
        if e == agent.start_epoch + 1:
            state = copy.deepcopy(init_state)
        else:
            state = env.reset()      # 获得当前的状态
            
        past_action = np.zeros(ACTION_DIMENSION)
        episode_reward_sum = 0       # 当前回合的奖励累计
        done = False                 # 是否回合结束
        episode_step = MAX_STEPS
        
        # 记录本回合的损失
        episode_critic1_losses = []
        episode_critic2_losses = []
        episode_actor_a_losses = []
        episode_actor_b_losses = []
        
        for t in range(episode_step):
            # ------------------------------------------------------------------
            # 动作选择：随机探索阶段或策略选择
            # ------------------------------------------------------------------
            if total_steps < RANDOM_ACTION_STEPS:
                # 随机探索阶段：前 RANDOM_ACTION_STEPS 步执行随机动作
                action = np.array([
                    np.random.uniform(RANDOM_V_MIN, ACTION_V_MAX),
                    np.random.uniform(-RANDOM_W_MAX, RANDOM_W_MAX)
                ], dtype=np.float32)
            else:
                # RAV-TD3 策略选择动作（双Actor投票）
                action = agent.choose_action(state, train=True)
                
            # ------------------------------------------------------------------
            # 执行动作并获取反馈
            # ------------------------------------------------------------------
            new_state, reward, done = env.step(action, past_action)

            # 异常奖励检测
            if (not np.isfinite(reward)) or abs(float(reward)) > 1e6:
                rospy.logwarn("Abnormal step reward detected: %s", str(reward))
                reward = -200.0
                done = True

            # 存储经验并执行训练
            agent.remember(state, action, reward, new_state, done)
            loss_dict = agent.learn()  # 执行 RAV-TD3 训练，返回损失
            
            # 记录损失
            if loss_dict is not None:
                episode_critic1_losses.append(loss_dict.get('critic1_loss', 0))
                episode_critic2_losses.append(loss_dict.get('critic2_loss', 0))
                if 'actor_a_loss' in loss_dict:
                    episode_actor_a_losses.append(loss_dict.get('actor_a_loss', 0))
                if 'actor_b_loss' in loss_dict:
                    episode_actor_b_losses.append(loss_dict.get('actor_b_loss', 0))
            
            episode_reward_sum += reward

            # 异常累计奖励检测
            if (not np.isfinite(episode_reward_sum)) or abs(float(episode_reward_sum)) > 1e6:
                rospy.logwarn("Abnormal episode reward sum detected: %s", str(episode_reward_sum))
                episode_reward_sum = REWARD_FAIL_LIMIT
                done = True

            total_steps += 1
            past_action = copy.deepcopy(action)
            state = copy.deepcopy(new_state)
            
            # 逐步日志
            if STEP_LOG and (t % STEP_LOG_INTERVAL == 0 or done):
                rospy.loginfo(
                    "ep=%d step=%d done=%s reward=%.2f reward_sum=%.2f v=%.3f w=%.3f",
                    e, t, str(done), float(reward), float(episode_reward_sum),
                    float(action[0]), float(action[1]),
                )

            # 超时检测
            if t >= MAX_STEPS - 1:
                rospy.loginfo("time out!")
                done = True
            
            # 奖励阈值检测
            if episode_reward_sum <= REWARD_FAIL_LIMIT:
                rospy.loginfo("reward fail")
                done = True

            if done:
                break

        # ===================================================================
        # 回合结束处理：保存奖励和损失
        # ===================================================================
        score_history.append(float(episode_reward_sum))
        with open(score_history_file, 'a') as f:
            f.write(str(float(episode_reward_sum)) + '\n')
        
        # 计算并记录本回合平均损失
        if episode_critic1_losses:
            c1_loss = float(np.mean(episode_critic1_losses))
            c2_loss = float(np.mean(episode_critic2_losses))
            a1_loss = float(np.mean(episode_actor_a_losses)) if episode_actor_a_losses else 0.0
            a2_loss = float(np.mean(episode_actor_b_losses)) if episode_actor_b_losses else 0.0
            loss_item = [e, c1_loss, c2_loss, a1_loss, a2_loss]
            loss_history.append(loss_item)

        # ===================================================================
        # 定期日志输出（每10回合）
        # ===================================================================
        if e % 10 == 0 and len(score_history) >= 10:
            avg10 = float(np.mean(score_history[-10:]))
            rospy.loginfo("Episode %d | avg10_score: %.2f | steps: %d", e, avg10, total_steps)
            
            # 输出最近损失
            if loss_history:
                last_loss = loss_history[-1]
                rospy.loginfo("  Losses | C1: %.4f | C2: %.4f | A1: %.4f | A2: %.4f",
                    last_loss[1], last_loss[2], last_loss[3], last_loss[4]
                )

        # ===================================================================
        # 最优模型追踪（按滑动平均奖励）
        # ===================================================================
        if len(score_history) >= MIN_EPISODES_FOR_BEST:
            cur_window = min(BEST_SCORE_WINDOW, len(score_history))
            cur_window_score = float(np.mean(score_history[-cur_window:]))
            if cur_window_score > (best_window_score + BEST_IMPROVE_DELTA):
                best_window_score = cur_window_score
                best_episode = int(e)
                no_improve_episodes = 0

                # 立即落盘当前最佳，避免后续回退导致最佳权重丢失
                agent.save_models(episode=e)
                with open(best_model_meta_file, 'w') as f:
                    json.dump({
                        'best_episode': best_episode,
                        'best_window_score': best_window_score,
                        'window': cur_window,
                        'updated_at_episode': int(e)
                    }, f, ensure_ascii=False, indent=2)
                rospy.loginfo("=== New Best Model | episode=%d | window_score=%.2f ===", best_episode, best_window_score)
            else:
                no_improve_episodes += 1

        # 可选早停：在长时间无提升时结束训练，减少后期策略漂移
        if EARLY_STOP_PATIENCE > 0 and no_improve_episodes >= EARLY_STOP_PATIENCE:
            rospy.loginfo("=== Early stop triggered | no_improve_episodes=%d ===", no_improve_episodes)
            break

        # ===================================================================
        # 定期保存（每SAVE_INTERVAL回合，默认60）
        # ===================================================================
        if e % SAVE_INTERVAL == 0:
            # 绘制奖励曲线
            filename_score = os.path.join(SAVE_FIGURE_PATH, "score_history_{}.png".format(e))
            plotLearning(score_history, filename_score, window=100)
            rospy.loginfo("Saved reward curve: %s", filename_score)
            
            # 保存损失记录（txt格式，可直接打开）
            with open(loss_history_file, 'w') as f:
                f.write("# Episode Critic1_Loss Critic2_Loss ActorA_Loss ActorB_Loss\n")
                for loss_item in loss_history:
                    f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        int(loss_item[0]), loss_item[1], loss_item[2], 
                        loss_item[3], loss_item[4]
                    ))
            rospy.loginfo("Saved loss history: %s", loss_history_file)
            
            # 保存模型权重（分文件夹）
            agent.save_models(episode=e)
            
            rospy.loginfo("=== Checkpoint saved at episode %d ===", e)
