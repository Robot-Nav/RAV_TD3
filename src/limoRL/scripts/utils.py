#!/usr/bin/env python3
"""
可视化与绘图工具模块
=====================
提供训练奖励曲线的绘制和保存功能。
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，支持无头服务器
import matplotlib.pyplot as plt
import numpy as np


def plotLearning(scores, filename, x=None, window=5):
    """
    绘制奖励学习曲线（带移动平均平滑）
    
    将原始奖励序列进行窗口平均处理，平滑噪声，
    生成奖励趋势曲线并保存为图片文件。
    
    Args:
        scores: 每轮奖励值列表（可能包含异常值）
        filename: 输出图片文件路径
        x: X 轴坐标（默认为 range(N)）
        window: 移动平均窗口大小（默认 5）
    """
    # 异常值清洗：NaN/Inf 替换为 0
    safe_scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    N = len(safe_scores)
    
    # 计算移动平均（窗口大小为 window）
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(safe_scores[max(0, t - window):(t + 1)])
    
    # 默认 X 轴为轮次索引
    if x is None:
        x = [i for i in range(N)]
    
    # 绘图
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')    # Y 轴：奖励值
    ax.set_xlabel('Step')     # X 轴：训练轮次
    ax.plot(x, running_avg)   # 绘制移动平均曲线
    fig.tight_layout()        # 自动调整布局
    fig.savefig(filename)     # 保存图片到文件
    plt.close(fig)            # 释放内存
