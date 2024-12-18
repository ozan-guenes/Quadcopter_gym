<div align="center"><h4> [AI611] Deep Reinforcement Learning - Final Report</h4>
<h1>Safety Aware UAV Navigation in Adverse Environments</h1>
<h4>Kasper Joergensen (20246358) & Ozan Günes (20225389)</h4>
</div>

---

## Project Motivation
The motivation for this project comes from the challenges of using reinforcement learning for drone navigation in real-world scenarios.​

Traditional drone control relies on classical controllers like PID, but these methods struggle in adverse environments with disturbances or obstacles.​

Reinforcement learning, particularly for continuous control tasks, offers the potential to adapt and optimize drone navigation policies autonomously.

### Challenges with Existing Methods:​
- **Handling Disturbances**: Many RL approaches struggle with unexpected disturbances
- **Safety in RL**: Safety measures in RL are still relatively underdeveloped

### Potential applications: ​
- Search-and-rescue missions in complex terrains.​
- Delivery in urban environments

## Simulation Environment 

## Challenges in Training RL Agent

## Reward Function Design 

## DDPG vs. TD3: Key Differences

## Performance Comparison: DDPG vs. TD3

## Reward Function Re-Design

## SAC and PPO: Key Differences

### Soft Actor-Critic (SAC):

SAC is an off-policy, model-free algorithm. It maximizes a trade-off between expected reward and entropy, which encourages exploration. This is achieved through a stochastic policy and the use of twin Q-networks to stabilize training. SAC also uses a replay buffer to sample past experiences efficiently.

#### Strengths of SAC:

- SAC is highly effective for continuous control tasks
- It has strong exploration capabilities due to entropy maximization
- It is computationally efficient compared to PPO, making it easier to scale

#### Challenges of SAC:

- SAC is computationally intensive because it trains twin Q-networks simultaneously.
- It requires careful fine-tuning of the entropy coefficient to balance exploration and exploitation.


| Feature | SAC | PPO |
| --- | --- | --- |
| Type | off-policy | on-policy |
| Exploration | strong (entropy driven) | moderate |
| Sample Efficiency | high (uses replay buffer) | lower (needs more samples) |
| Training Stability | moderate (requires careful tuning) | high |
| Performance | better for continuous control | balanced for all tasks |


## Performance Comparison: SAC and PPO

## Conclusions and Limitations
