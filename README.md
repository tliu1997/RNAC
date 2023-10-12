## **Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation**
Implementation of the algorithm Robust Natural Actor Critic (RNAC). RNAC is introduced in our paper [Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation (NeurIPS 2023)](https://arxiv.org/abs/2307.08875). This implementation of RNAC is based on the implementation of [Proximal Policy Optimization (PPO)](https://github.com/Lizhi-sjtu/DRL-code-pytorch.git), which includes several tricks to make its performance comparable to [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290).

RNAC is tested in three MuJoCo continuous control tasks (i.e., `Hopper-v3`, `HalfCheetah-v3`, and `Walker-v3`) and a real-world `TurtleBot` navigation task [Video of TurtleBot Demonstration](https://www.youtube.com/playlist?list=PLO9qoak6TW3EZ_UrfADvKvQCITewIBOxG). To verify the robust performance of algorithms, several perturbed MuJoCo environments are created based on [the previous implementation](https://github.com/zaiyan-x/RFQI.git). Note that MuJoCo should be installed before using this repo.

We propose two novel uncertainty sets, i.e., double-sampling (DS) uncertainty set and integral probability metric (IPM) uncertainty set, which makes large-scale robust RL tractable even when one only has access to a simulator. IPM uncertainty set works for both deterministic and stochastic models, while DS uncertainty set is only reasonable for stochastic models. Since MuJoCo envs are deterministic, we add a uniform actuation noise ~ Unif[-5e-3, 5e-3] in constructing stochastic MuJoCo envs. DS uncertainty set is tested in stochastic envs (self.do_simulation(action_noise, self.frame_skip) in both original envs and perturbed envs), while IPM uncertainty set is tested in deterministic envs (self.do_simulation(action, self.frame_skip) in both original envs and perturbed envs).


### Prerequisites
Here we list our running environment:
- gym == 0.21.0
- mujoco-py == 2.1.2.14
- PyTorch == 1.13.1
- numpy == 1.24.1
- matplotlib == 3.4.3


We then need to properly register the perturbed Gym environments within the folder perturned_env.
1. Add hopper_perturbed.py, half_cheetach_perturbed.py, and walker2d_perturbed.py under gym/envs/mujoco
2. Replace mujoco_env.py, hopper_v3.py, half_cheetah_v3.py, and walker2d_v3.py under gym/envs/mujoco
3. Add the following to _init_.py under gym/envs:
```
register(
    id="HopperPerturbed-v3",
    entry_point="gym.envs.mujoco.hopper_perturbed:HopperPerturbedEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="HalfCheetahPerturbed-v3",
    entry_point="gym.envs.mujoco.half_cheetah_perturbed:HalfCheetahPerturbedEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id="Walker2dPerturbed-v3",
    max_episode_steps=1000,
    entry_point="gym.envs.mujoco.walker2d_perturbed:Walker2dPerturbedEnv",
)
```



### Instruction
Here we use `Hopper-v3` as an example.

#### Train Policy
To train an RNAC policy on `Hopper-v3` with DS uncertainty set, please run 
```
python train_rnac.py --env='Hopper-v3' --uncer_set='DS' --weight_reg=0.0
```

To train an RNAC policy on `Hopper-v3` with IPM uncertainty set, please run 
```
python train_rnac.py --env='Hopper-v3' --uncer_set='IPM' --weight_reg=1e-5
```

#### Evaluate Policy
To evaluate an RNAC policy on `Hopper-v3`, please run
```
python eval_rnac.py --env='Hopper-v3'
```