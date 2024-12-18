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


### Proximal Policy Optimization (PPO):
PPO, on the other hand, is an on-policy, model-free algorithm. It optimizes a clipped surrogate objective, which prevents large, unstable updates to the policy. PPO uses single policy networks and avoids replay buffers, making it simpler and more stable.

#### Strengths of PPO:
- PPO is simpler to implement and produces more stable training results
- It is suitable for environments with either discrete or continuous actions
- PPO is popular for training agents in large-scale distributed systems

#### Challenges of PPO:
- PPO is less sample-efficient compared to off-policy methods like SAC. It requires more interactions with the environment, which can be costly.
   
<div align="center">

| Feature | SAC | PPO |
| --- | --- | --- |
| Type | off-policy | on-policy |
| Exploration | strong (entropy driven) | moderate |
| Sample Efficiency | high (uses replay buffer) | lower (needs more samples) |
| Training Stability | moderate (requires careful tuning) | high |
| Performance | better for continuous control | balanced for all tasks |

</div>


## Performance Comparison: SAC and PPO

## Conclusions and Limitations
