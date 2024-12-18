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

In this project, we utilized a drone simulation environment based on standard quadcopter dynamics, defined by ordinary differential equations. The dynamics were derived from _Quadcopter Dynamics, Simulation, and Control_ by Andrew Gibiansky and _Quadrotor Dynamics and Control_ by Randal Beard. We employed an implementation of these dynamics from [this GitHub repository](https://github.com/abhijitmajumdar/Quadcopter_simulator), which includes a PID controller for quadcopter control and navigation.

The drone's state is represented by a 12-dimensional vector comprising:

- Position: \([x, y, z]\),
- Linear velocities,
- Angular orientations: \([\theta, \phi, \psi]\),
- Angular rates.

The action space includes thrust forces from the four motors, each ranging from 4000 to 9000 units, simulating realistic motor inputs. To facilitate reinforcement learning, the state space was extended to include a goal position and a goal yaw orientation, forming a 16-dimensional vector that encompasses both the drone's state and the goal state.

<div align="center">
    <img src="./figures/quadcopter_env.png" alt="Quadcopter Environment" width="70%">
</div>

<div align="center">
    <img src="./figures/goal_state.png" alt="Goal State Representation" width="35%">
</div>

Each simulation episode begins by randomly sampling the drone's initial and goal states. The task is for the drone to navigate to the target, at which point a new goal is generated.

## Challenges in Training RL Agent

## Reward Function Design 

## DDPG vs. TD3: Key Differences

## Performance Comparison: DDPG vs. TD3

The DDPG algorithm failed to converge to a meaningful policy during training, highlighting its inability to effectively navigate the quadcopter environment. The TD3 algorithm showed improvements over DDPG, but performance remained suboptimal.

When evaluating the trained agents from both methods over 100 episodes, the mean rewards and standard deviations were as follows:

- **DDPG**: Mean reward of \(-2091 \pm 1198\)
- **TD3**: Mean reward of \(-1588 \pm 837\)

Both algorithms exhibited high variance, and although TD3 outperformed DDPG, the results suggest that both approaches achieved suboptimal rewards. Qualitative simulations further revealed only slight improvements with TD3 over an untrained agent. Simulations of the untrained TD3 agent (left) and trained agent (right) are shown below, demonstrating the limited gains achieved.

|                 TD3 Untrained                 |                TD3 Trained                |          DDPG vs TD3 Comparison           |
| :-------------------------------------------: | :---------------------------------------: | :---------------------------------------: |
| ![TD3 Untrained](./figures/td3_untrained.gif) | ![TD3 Trained](./figures/td3_trained.gif) | ![DDPG vs TD3](./figures/ddpg_vs_td3.png) |

These findings indicate that while TD3 addressed some shortcomings, it still failed to deliver reliable drone navigation. To address this, we simplified the learning environment by fixing the drone's start and goal positions across all episodes, aiming to make the policy easier to learn. Additionally, we tested the SAC and PPO algorithms within this simplified environment to explore alternative approaches.

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

| Feature            | SAC                                | PPO                        |
| ------------------ | ---------------------------------- | -------------------------- |
| Type               | off-policy                         | on-policy                  |
| Exploration        | strong (entropy driven)            | moderate                   |
| Sample Efficiency  | high (uses replay buffer)          | lower (needs more samples) |
| Training Stability | moderate (requires careful tuning) | high                       |
| Performance        | better for continuous control      | balanced for all tasks     |

## Performance Comparison: SAC and PPO

## Conclusions and Limitations
