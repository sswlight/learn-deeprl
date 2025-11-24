## 代码实现
`train.py`:实际训练
`test.py`:gym的可视化
## deeprl几种baseline的比较


### 深度强化学习核心算法对比表

| 维度 | DQN (Deep Q-Network) | A2C (Advantage Actor-Critic) | PPO (Proximal Policy Optimization) | TD3 (Twin Delayed DDPG) | SAC (Soft Actor-Critic) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **类型** | Value-based / Off-Policy | Actor-Critic / On-Policy | Actor-Critic / On-Policy | Actor-Critic / Off-Policy | Actor-Critic / Off-Policy |
| **动作空间** | **离散** (Discrete) | 离散 / 连续 | 离散 / 连续 | **连续** (Continuous) | **连续** (Continuous) |
| **稳定性** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **样本效率** | ⭐⭐⭐⭐ (中高) | ⭐⭐ (低) | ⭐⭐⭐ (中) | ⭐⭐⭐⭐⭐ (高) | ⭐⭐⭐⭐⭐ (极高) |
| **资源占用** | **高** (Replay Buffer) | **低** | **中** (Rollout Buffer) | **高** (Buffer + 2 Critics) | **高** (Buffer + 2 Critics + Alpha) |
| **核心 Trick** | 1. Experience Replay<br>2. Target Network | 1. Parallel Env<br>2. N-step Return | 1. Ratio Clipping<br>2. GAE<br>3. Advantage Norm | 1. Twin Critics<br>2. Delayed Update<br>3. Target Noise | 1. Max Entropy<br>2. Reparameterization<br>3. Auto Alpha |
| **探索方式** | $\epsilon$-Greedy (概率随机) | 熵正则化 (Entropy Bonus) | 熵正则化 (Entropy Bonus) | 动作加噪 $\mathcal{N}(0, \sigma)$ | **最大熵** (策略自带随机性) |
| **Actor 训练** | **无**<br>(由 Q 值选 Max) | **随机策略梯度**<br>$\nabla \log \pi \cdot A$ | **截断策略梯度**<br>限制比率 $r(\theta)$ | **确定性策略梯度**<br>$\nabla_a Q \cdot \nabla_\theta \mu$ | **重参数化技巧**<br>最大化 $Q + \alpha H$ |
| **Critic 训练** | Regression<br>拟合 Bellman Target | Regression<br>拟合 $r + \gamma V'$ | Regression<br>拟合 GAE Returns | Regression<br>拟合 Twin Target | Regression<br>拟合 Soft Twin Target |
| **Actor Loss** | N/A | $L = - \mathbb{E} [\log \pi(a\|s) \cdot A_t]$ | $L = - \mathbb{E} [\min(r_t A_t, \text{clip}(r_t) A_t)]$ | $L = - \mathbb{E} [Q_1(s, \mu(s))]$ | $L = \mathbb{E} [\alpha \log \pi(a\|s) - Q_1(s, a)]$ |
| **Critic Loss** | $L = (y - Q)^2$<br>Target: $y = r + \gamma \max Q'$ | $L = (y - V)^2$<br>Target: $y = r + \gamma V'$ | $L = (R_t - V)^2$<br>Target: $R_t = A_t + V$ | $L = \sum (y - Q_i)^2$<br>Target: $y = r + \gamma \min Q'_i$ | $L = \sum (y - Q_i)^2$<br>Target: $y = r + \gamma (Q'_{min} - \alpha \log \pi')$ |
| **Baseline** | **不需要** | **需要** (V值) | **需要** (V值) | **不需要** | **不需要** |

---

### 公式符号说明：

*   $s, a, r, s'$: 当前状态、动作、奖励、下一状态。
*   $\gamma$: 折扣因子 (Gamma)。
*   $A_t$: 优势函数 (Advantage)。
*   $\pi(a|s)$: 随机策略（Actor 输出概率）。
*   $\mu(s)$: 确定性策略（Actor 输出动作值）。
*   $r_t(\theta)$: 新旧策略比率 $\frac{\pi_{new}}{\pi_{old}}$。
*   $Q'$: 目标网络 (Target Network) 的输出。
*   $\alpha$: SAC 中的温度系数 (Entropy Coefficient)。
*   $H$: 熵 (Entropy)，$H = -\log \pi(a|s)$。

## 不足
- 根据StableBaseline3完成代码，但是并没有采用“继承基类”并统一文件的方式，而是将不同的算法分别实现
- 没有统一参数，神经网络神经元与训练次数不同，没有控制变量实现对比
