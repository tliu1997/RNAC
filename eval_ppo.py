import os
import numpy as np
import gym
import argparse
import pickle
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous


def evaluate_policy(args, env, agent, state_norm, springref, x):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset(state=None, x_pos=None, springref=springref, leg_joint_stiffness=x)
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
    

def load_agent(agent, load_path, device):
    agent.actor.load(f'{load_path}_actor', device=device)
    agent.critic.load(f'{load_path}_critic', device=device)
    with open(f'{load_path}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)
    with open(f'{load_path}_reward_scaling', 'rb') as file2:
        reward_scaling = pickle.load(file2)

    return agent, state_norm, reward_scaling


def save_evals(save_path, setting, avgs, stds, GAMMA):
    np.save(f'{save_path}_{setting}_avgs_{GAMMA}', avgs)
    np.save(f'{save_path}_{setting}_stds_{GAMMA}', stds)


def main(args):
    seed, GAMMA = args.seed, args.GAMMA
    # evaluate PPO on perturbed environments
    load_path = f"./models/PPO_{args.env}_{GAMMA}"
    save_path = f"./perturbed_results/PPO_{args.env}_{GAMMA}"

    # get perturbed environment
    i = args.env.find('-')
    perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'
    env = gym.make(perturbed_env)
    env_evaluate = gym.make(perturbed_env)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    agent = PPO_continuous(args)
    agent, state_norm, reward_scaling = load_agent(agent, load_path, args.device)
    agent.gamma = args.gamma

    eval_episodes = args.eval_episodes
    if args.hard == 'False':
        hard = False
    else:
        hard = True

    if args.env == 'Hopper-v3':
        # settings = ['leg_joint_stiffness', 'gravity', 'joint_damping']
        springref = 2.0

        xs_legjntstif = np.arange(0.0, 60.0, 5.0)
        #ps_g = np.arange(-0.5, 0.05, 0.05)
        #ps_damp = np.arange(0.0, 1.1, 0.1)

        setting = 'leg_joint_stiffness'
        #setting = 'gravity'
        #setting = 'joint_damping'
        
        avgs = []
        stds = []
        for x in xs_legjntstif:
        #for x in ps_g:
        #for x in ps_damp:
            rewards = []
            for _ in range(args.eval_episodes):
                env.seed(seed=np.random.randint(1000))
                evaluate_reward = evaluate_policy(args, env, agent, state_norm, springref, x)
                rewards.append(evaluate_reward)
            
            avg_reward = np.sum(rewards) / args.eval_episodes
            print("---------------------------------------")
            print(f' leg joint stiffness with x {x}: {avg_reward:.3f}')
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds, GAMMA)

    if args.env == 'Walker2d-v3':
        # settings = ['foot_joint_stiffness', 'leg_joint_stiffness']
        xs_ftjntstif = np.arange(0.0, 50.0, 5.0)
        #xs_legjntstif = np.arange(0.0, 60.0, 5.0)
        
        setting = 'foot_joint_stiffness'
        #setting = 'leg_joint_stiffness'

        avgs = []
        stds = []
        for x in xs_ftjntstif:
        #for x in xs_legjntstif:
            rewards = []
            for _ in range(args.eval_episodes):
                env.seed(seed=np.random.randint(1000))
                evaluate_reward = evaluate_policy(args, env, agent, state_norm, x)
                rewards.append(evaluate_reward)
            
            avg_reward = np.sum(rewards) / args.eval_episodes
            print("---------------------------------------")
            print(f' leg joint stiffness with x {x}: {avg_reward:.3f}')
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds, GAMMA)

    if args.env == 'HalfCheetah-v3':
        #xs_bact = np.arange(0.3, 1.1, 0.1)
        ps_fjntstif = np.arange(0.0, 0.80, 0.05)

        #setting = 'back_actuator_ctrlrange'
        setting = 'front_joint_stiffness'

        avgs = []
        stds = []
        #for p in xs_bact:
        for p in ps_fjntstif:        
            rewards = []
            env.reset(state=None, x_pos=None)
            fthigh_joint_stiffness = env.fthigh_joint_stiffness * (1 + p)
            fshin_joint_stiffness = env.fshin_joint_stiffness * (1 + p)
            ffoot_joint_stiffness = env.ffoot_joint_stiffness * (1 + p)
            for _ in range(eval_episodes):
                env.seed(seed=np.random.randint(1000))
                #evaluate_reward = evaluate_policy(args, env, agent, state_norm, p)
                evaluate_reward = evaluate_policy(args, env, agent, state_norm, fthigh_joint_stiffness, fshin_joint_stiffness, ffoot_joint_stiffness)
                rewards.append(evaluate_reward)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' front joints stiffness with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds, GAMMA)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument('--hard', default='False', type=str)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument("--env", type=str, default='Hopper-v3', help="HalfCheetah-v3/Hopper-v3/Walker2d-v3")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
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
    parser.add_argument("--weight_reg", type=float, default=1e-5, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="Seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="Name")

    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")

    main(args)
