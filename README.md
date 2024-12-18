<div align="center"><h4> [AI611] Deep Reinforcement Learning - Final Report</h4>
<h1>Safety Aware UAV Navigation in Adverse Environments</h1>
<h4>Kasper Joergensen (20246358) & Ozan Günes (20225389)</h4>
</div>

---

## Project Motivation

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

<div style="display: flex; justify-content: space-between; align-items: center;">

  <figure style="text-align: center; width: 32%;">
    <img src="./figures/td3_untrained.gif" alt="TD3 Untrained" style="width: 100%; height: auto;">
    <figcaption>TD3 Untrained</figcaption>
  </figure>
  
  <figure style="text-align: center; width: 32%;">
    <img src="./figures/td3_trained.gif" alt="TD3 Trained" style="width: 100%; height: auto;">
    <figcaption>TD3 Trained</figcaption>
  </figure>
  
  <figure style="text-align: center; width: 32%;">
    <img src="./figures/ddpg_vs_td3.png" alt="DDPG vs TD3" style="width: 100%; height: auto;">
    <figcaption>DDPG vs TD3 Comparison</figcaption>
  </figure>

</div>
These findings indicate that while TD3 addressed some shortcomings, it still failed to deliver reliable drone navigation. To address this, we simplified the learning environment by fixing the drone's start and goal positions across all episodes, aiming to make the policy easier to learn. Additionally, we tested the SAC and PPO algorithms within this simplified environment to explore alternative approaches.

## Reward Function Re-Design

## SAC and PPO: Key Differences

## Performance Comparison: SAC and PPO

## Conclusions and Limitations
