import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
import pickle
import math
import random
import copy
import mujoco_py
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from torch.distributions import Uniform


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset(state=None, x_pos=None)
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def save_agent(agent, save_path, state_norm, reward_scaling):
    agent.actor.save(f'{save_path}_actor')
    agent.critic.save(f'{save_path}_critic')
    with open(f'{save_path}_state_norm', 'wb') as file1:
        pickle.dump(state_norm, file1)
    with open(f'{save_path}_reward_scaling', 'wb') as file2:
        pickle.dump(reward_scaling, file2)


def main(args, number):
    seed, GAMMA = args.seed, args.GAMMA
    env = gym.make(args.env)
    env_evaluate = gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
    env_reset = gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    env_reset.seed(seed)
    env_reset.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    max_value = -np.inf
    save_path = f"./models/RNAC_{args.env}_{GAMMA}"

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/RNAC/env_{}_{}_number_{}_seed_{}_GAMMA_{}'.format(args.env, args.policy_dist, number, seed, GAMMA))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        #if total_steps > args.max_train_steps // 2:
        #    agent.gamma = 0.999
        s = env.reset(state=None, x_pos=None)
        s_org, x_pos = copy.deepcopy(s), np.array([env.sim.data.qpos[0]])
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            #if total_steps < args.random_steps:  # Take the random actions in the beginning for the better exploration
            #    a = env.action_space.sample()
            #    s_tensor = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
            #    with torch.no_grad():
            #        dist = agent.actor.get_dist(s_tensor)
            #        a_logprob = dist.log_prob(torch.Tensor(a)).numpy().flatten()
            #else:
            #    a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            
            if args.uncer_set == "DS":
                # Multi-run
                v_min, index = torch.tensor(float('inf')), 0
                noise_list, nexts_list, r_list = [], [], []
                for i in range(args.next_steps):
                    obs = env_reset.reset(state=s_org, x_pos=x_pos) 
                    s_, r, done, info = env_reset.step(action)
                    r_list.append(r)
                    noise_list.append(info['noise'])
                    if args.use_state_norm:
                        s_ = state_norm(s_, update=False)
                    nexts_list.append(s_)
                    with torch.no_grad():
                        if agent.critic(torch.tensor(s_, dtype=torch.float)) < v_min:
                            v_min = agent.critic(torch.tensor(s_, dtype=torch.float))
                            index = i
            
                # pick next state for robust critic update
                ridx = random.randint(0, args.next_steps)
                if ridx == args.next_steps:
                    ridx = index
                s_, r, done, info = env.step(np.concatenate((action, noise_list[ridx])))
            else:
                s_, r, done, info = env.step(action)
            x_pos = np.array([info['x_position']])
            if args.use_state_norm:
                #nexts = state_norm(nexts, update=False)
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = copy.deepcopy(s_)
            s_org = copy.deepcopy(state_norm.denormal(s_, update=False))
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/RNAC_{}_env_{}_number_{}_seed_{}_GAMMA_{}.npy'.format(args.policy_dist, args.env, number, seed, GAMMA), np.array(evaluate_rewards))

                # save actor, critic for evaluation in perturbed environment
                if evaluate_reward > max_value:
                    save_agent(agent, save_path, state_norm, reward_scaling)
                    max_value = evaluate_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='Hopper-v3', help="HalfCheetah-v3/Hopper-v3/Walker2d-v3")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")

    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    main(args, number=1)
