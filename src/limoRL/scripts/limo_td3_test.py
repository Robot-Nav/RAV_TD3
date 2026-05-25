#!/usr/bin/env python3
"""
TD3 模型测试主程序
===================
加载训练好的 TD3 模型并在仿真环境中执行评估测试。

测试流程：
1. 自动扫描检查点目录，选择最新完整模型
2. 执行多个测试 episode
3. 统计成功率、碰撞率、平均奖励
"""

import copy
import glob
import json
import math
import os
import sys

# 确保当前脚本目录在 Python 路径中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import rospy

from TD3.TD3Net import TD3
from TD3.Environment import Env

# 完整模型检查点所需的文件类型（不含后缀）
REQUIRED_CKPT_KEYS = [
    'actor',              # 主策略网络 A
    'target_actor',       # 目标策略网络 A
    'critic1',            # 价值网络 1
    'target_critic1',     # 目标价值网络 1
    'critic2',            # 价值网络 2
    'target_critic2',     # 目标价值网络 2
]


def _has_full_checkpoint_set(ckpt_dir, episode):
    """
    检查指定轮次是否存在完整的模型文件集合
    
    检查子文件夹结构：
        actor/actor_{episode}.pth
        target/target_actor_{episode}.pth
        target/target_actor_b_{episode}.pth
        critic1/critic1_{episode}.pth
        target/target_critic1_{episode}.pth
        critic2/critic2_{episode}.pth
        target/target_critic2_{episode}.pth
    
    Args:
        ckpt_dir: 模型检查点目录
        episode: 训练轮次编号
        
    Returns:
        bool: True 表示所有必需模型文件都存在
    """
    required_files = [
        os.path.join(ckpt_dir, 'actor', 'actor_{}.pth'.format(episode)),
        os.path.join(ckpt_dir, 'target', 'target_actor_{}.pth'.format(episode)),
        os.path.join(ckpt_dir, 'target', 'target_actor_b_{}.pth'.format(episode)),
        os.path.join(ckpt_dir, 'critic1', 'critic1_{}.pth'.format(episode)),
        os.path.join(ckpt_dir, 'target', 'target_critic1_{}.pth'.format(episode)),
        os.path.join(ckpt_dir, 'critic2', 'critic2_{}.pth'.format(episode)),
        os.path.join(ckpt_dir, 'target', 'target_critic2_{}.pth'.format(episode)),
    ]
    for ckpt in required_files:
        if not os.path.isfile(ckpt):
            return False
    return True


def _list_actor_episodes(ckpt_dir):
    """
    扫描 actor 子文件夹中所有可用的模型轮次编号
    
    Args:
        ckpt_dir: 模型检查点目录
        
    Returns:
        list: 排序后的轮次编号列表
    """
    episodes = []
    actor_dir = os.path.join(ckpt_dir, 'actor')
    if not os.path.isdir(actor_dir):
        return []
    for ckpt in glob.glob(os.path.join(actor_dir, 'actor_*.pth')):
        base = os.path.basename(ckpt)
        token = base.replace('actor_', '').replace('.pth', '')
        try:
            episodes.append(int(token))
        except ValueError:
            continue
    return sorted(set(episodes))


def _resolve_best_episode_from_meta(ckpt_dir):
    """
    从训练阶段生成的 best_model.json 中读取最佳轮次

    Args:
        ckpt_dir: 模型检查点目录

    Returns:
        int or None: 最佳轮次，若无效则返回 None
    """
    meta_file = os.path.join(ckpt_dir, 'best_model.json')
    if not os.path.isfile(meta_file):
        return None

    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        best_ep = int(meta.get('best_episode', -1))
        if best_ep >= 0:
            return best_ep
    except Exception as e:
        rospy.logwarn('Failed to parse %s: %s', meta_file, str(e))

    return None


def resolve_model_episode(ckpt_dir, requested_episode, select_mode='best'):
    """
    解析要加载的模型轮次编号
    
    选择逻辑：
    1. 如果指定了有效轮次且模型完整，使用指定轮次
     2. 若未指定轮次：
         - select_mode=best：优先读取 best_model.json
         - select_mode=latest：加载最新完整模型
         - select_mode=auto：先best后latest
    
    Args:
        ckpt_dir: 模型检查点目录
        requested_episode: 请求的轮次编号（-1 表示自动选择）
        select_mode: 模型选择模式（best/latest/auto）
        
    Returns:
        int: 实际使用的模型轮次编号
        
    Raises:
        RuntimeError: 未找到可用模型时抛出
    """
    actor_episodes = _list_actor_episodes(ckpt_dir)
    if not actor_episodes:
        raise RuntimeError('No TD3 actor checkpoints found in {}'.format(ckpt_dir))

    # 过滤出完整模型集
    valid_episodes = [ep for ep in actor_episodes if _has_full_checkpoint_set(ckpt_dir, ep)]
    if not valid_episodes:
        raise RuntimeError('No complete TD3 checkpoint set found in {}'.format(ckpt_dir))

    # 用户指定了有效轮次
    if requested_episode >= 0 and requested_episode in valid_episodes:
        return requested_episode

    # 用户指定的轮次不存在或不完整，回退到最新
    if requested_episode >= 0:
        rospy.logwarn(
            'Requested episode %d is missing or incomplete, fallback to latest valid episode %d',
            requested_episode,
            valid_episodes[-1],
        )

    mode = str(select_mode).lower().strip()
    if mode not in ('best', 'latest', 'auto'):
        mode = 'best'

    if requested_episode < 0 and mode in ('best', 'auto'):
        best_ep = _resolve_best_episode_from_meta(ckpt_dir)
        if best_ep is not None and best_ep in valid_episodes:
            rospy.loginfo('Using best episode from metadata: %d', best_ep)
            return best_ep
        if mode == 'best':
            rospy.logwarn('Best model metadata unavailable or invalid, fallback to latest valid episode %d', valid_episodes[-1])

    return valid_episodes[-1]  # 返回最新完整模型


if __name__ == '__main__':
    ACTION_DIMENSION = 2                                    # 动作空间维度
    ACTION_V_MAX = float(rospy.get_param('~action_v_max', 0.6))   # 线速度上限 (m/s)
    ACTION_W_MAX = float(rospy.get_param('~action_w_max', 1.0))   # 角速度上限 (rad/s)
    CKPT_DIR = "/home/zjs/DRL_ws/src/limoRL/train/TD3/model"      # 模型保存路径

    # 初始化 ROS 节点
    rospy.init_node('limo_td3_test')

    # 从参数服务器读取测试配置
    test_episode_count = rospy.get_param('~test_episodes', 20)           # 测试 episode 总数
    max_steps = rospy.get_param('~max_steps', 1200)                      # 每 episode 最大步数
    model_episode = int(rospy.get_param('~episode', -1))                 # 指定加载的模型轮次（-1 = 自动）
    model_select_mode = str(rospy.get_param('~model_select_mode', 'best'))  # 自动模式下模型选择策略
    success_reward_threshold = rospy.get_param('~success_reward_threshold', 250.0)  # 成功判定奖励阈值
    alpha = float(rospy.get_param('~alpha', 0.0001))                     # Actor 学习率（创建智能体时需要）
    beta = float(rospy.get_param('~beta', 0.0001))                       # Critic 学习率
    actor_fc1_dim = int(rospy.get_param('~actor_fc1_dim', 512))          # Actor 网络维度
    actor_fc2_dim = int(rospy.get_param('~actor_fc2_dim', 256))
    critic_fc1_dim = int(rospy.get_param('~critic_fc1_dim', 512))        # Critic 网络维度
    critic_fc2_dim = int(rospy.get_param('~critic_fc2_dim', 256))

    # 创建环境
    env = Env(action_dim=ACTION_DIMENSION)
    try:
        init_state = env.reset()
    except rospy.ROSException as e:
        rospy.logerr('Env reset failed: %s', str(e))
        rospy.logerr('Please launch simulator first: roslaunch limo_gazebo_sim limoEnv.launch')
        sys.exit(1)

    state_dim = int(init_state.shape[0])
    rospy.loginfo('Detected TD3 state_dim: %d', state_dim)

    # 创建智能体（测试模式，学习率等参数不影响推理）
    agent = TD3(
        alpha=alpha,
        beta=beta,
        state_dim=state_dim,
        action_dim=2,
        actor_fc1_dim=actor_fc1_dim,
        actor_fc2_dim=actor_fc2_dim,
        action_limit_v=ACTION_V_MAX,
        action_limit_w=ACTION_W_MAX,
        critic_fc1_dim=critic_fc1_dim,
        critic_fc2_dim=critic_fc2_dim,
        ckpt_dir=CKPT_DIR,
        gamma=0.99,
        tau=0.005,
        action_noise=0.03,         # 测试模式使用较小的噪声
        policy_noise=0.08,
        policy_noise_clip=0.3,
        delay_time=2,
        max_size=1000,
        batch_size=2,
    )

    # 解析并加载模型
    try:
        load_episode = resolve_model_episode(CKPT_DIR, model_episode, model_select_mode)
    except RuntimeError as e:
        rospy.logerr(str(e))
        sys.exit(1)

    rospy.loginfo('Loading TD3 model from episode: %d', load_episode)
    try:
        agent.load_models(episode=load_episode)
    except Exception as e:
        rospy.logerr('Load model failed for episode %d: %s', load_episode, str(e))
        sys.exit(1)

    # 测试统计变量
    rewards = []
    success_count = 0
    collision_count = 0

    # 执行测试循环
    for e in range(test_episode_count):
        if e == 0:
            state = copy.deepcopy(init_state)
        else:
            state = env.reset()
        past_action = np.zeros(ACTION_DIMENSION)
        episode_reward_sum = 0.0
        done = False
        success = False

        for t in range(max_steps):
            # 测试模式：不添加探索噪声
            action = agent.choose_action(state, train=False)
            new_state, reward, done = env.step(action, past_action)

            episode_reward_sum += reward
            past_action = copy.deepcopy(action)
            state = copy.deepcopy(new_state)

            # 成功判定：奖励达到阈值
            if reward >= success_reward_threshold:
                success = True
                done = True

            # 提前终止：累计奖励过低
            if episode_reward_sum <= -2400:
                rospy.loginfo('Episode %d early stop: reward fail', e)
                done = True

            if done:
                break

        # 统计结果
        if success:
            success_count += 1
        elif done:
            collision_count += 1

        rewards.append(episode_reward_sum)
        rospy.loginfo(
            'Test Episode %d | steps=%d | reward=%.2f | success=%s',
            e,
            t + 1,
            episode_reward_sum,
            str(success),
        )

    # 计算汇总统计
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    success_rate = (success_count / float(test_episode_count)) * 100.0 if test_episode_count > 0 else 0.0

    # 打印测试报告
    rospy.loginfo('================Improve TD3 Test Summary ================')
    rospy.loginfo('Model episode: %d', load_episode)
    rospy.loginfo('Total test episodes: %d', test_episode_count)
    rospy.loginfo('Average reward: %.2f', avg_reward)
    rospy.loginfo('Success count: %d', success_count)
    rospy.loginfo('Collision/failed count: %d', collision_count)
    rospy.loginfo('Success rate: %.2f%%', success_rate)
