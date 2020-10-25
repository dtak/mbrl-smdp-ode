import os
import argparse
import random
import time
import numpy as np
import torch
from mbrl import MBRL
import utils

from envs.windygrid_simulator import WindyGridSimulator
from envs.hiv_simulator import HIVSimulator
from envs.acrobot_simulator import AcrobotSimulator
try:
    from envs.half_cheetah_simulator import HalfCheetahSimulator
    from envs.swimmer_simulator import SwimmerSimulator
    from envs.hopper_simulator import HopperSimulator
except:
    print("Couldn't import Mujoco.")

parser = argparse.ArgumentParser('Running model-based RL')
parser.add_argument('--train_env_model', action='store_true', help='train environment model')
parser.add_argument('--world_model', action='store_true', help='learn the world model')
parser.add_argument('--latent_policy', action='store_true', help='whether make decision based on latent variables')
parser.add_argument('--num_restarts', type=int, default=0, help='the number of restarts')
parser.add_argument('--model', type=str, default='free', help='the environment model, or load the training model')
parser.add_argument('--trained_model_path', type=str, default='', help='the pre-trained environment model path')
parser.add_argument('--env', type=str, default='acrobot', help='the environment')
parser.add_argument('--timer', type=str, default='fool', help='the type of timer')
parser.add_argument('--seed', type=int, default=2020, help='the random seed')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
parser.add_argument('--obs_normal', action='store_true', help='whether normalize the observation')
parser.add_argument('--latent_dim', type=int, default=10, help='the latent state dimension')
parser.add_argument('--ode_tol', type=float, default=1e-3, help='the relative error tolerance of ODE networks')
parser.add_argument('--ode_dim', type=int, default=5, help='the number of hidden units in ODE network')
parser.add_argument('--enc_hidden_to_latent_dim', type=int, default=5, help='the number of hidden units for hidden to latent')
parser.add_argument('--lr', type=float, default=9e-4, help='the learning rate for training environment model')
parser.add_argument('--batch_size', type=int, default=32, help='the mini-batch size for training environment model')
parser.add_argument('--epochs', type=int, default=150, help='the number of epochs for training environment model')
parser.add_argument('--iters', type=int, default=12000, help='the number of iterations for training environment model')
parser.add_argument('--trajs', type=int, default=1000, help='the number of trajs for training environment model')
parser.add_argument('--eps_decay', type=float, default=1e-4, help='the linear decay rate for scheduled sampling')
parser.add_argument('--max_steps', type=int, help='the max steps for running policy and trajectory generation')
parser.add_argument('--episodes', type=int, default=1000, help='the number of episodes for running policy')
parser.add_argument('--mem_size', type=int, default=int(1e5), help='the size of experience replay buffer')
parser.add_argument('--log', action='store_true', help='using logger or print')
parser.add_argument('--mpc_ac', action='store_true', help='model predictive control for actor-critic')
parser.add_argument('--mb_epochs', type=int, default=10, help='the epochs for iterative training')
parser.add_argument('--mf_epochs', type=int, default=240, help='the epochs for iterative training')
parser.add_argument('--planning_horizon', type=int, default=15, help='the planning horizon for environment model')
parser.add_argument('--env_steps', type=int, default=4000, help='the number of environment steps per epoch')
args = parser.parse_args()

if not os.path.exists("models/"):
    utils.makedirs("models/")
if not os.path.exists("logs/"):
    utils.makedirs("logs/")
if not os.path.exists("results/"):
    utils.makedirs("results/")

# seed for reproducibility
exp_id = int(random.SystemRandom().random() * 100000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if args.env == 'grid':
    simulator = WindyGridSimulator()
elif args.env == 'acrobot':
    simulator = AcrobotSimulator()
elif args.env == 'hiv':
    simulator = HIVSimulator()
elif args.env == 'hiv-pomdp':
    simulator = HIVSimulator(podmp=True)
elif args.env == 'half_cheetah':
    simulator = HalfCheetahSimulator()
elif args.env == 'swimmer':
    simulator = SwimmerSimulator()
elif args.env == 'hopper':
    simulator = HopperSimulator()
else:
    raise NotImplementedError
simulator.seed(args.seed)

ckpt_path = 'models/{}_{}_{}.ckpt'.format(args.model, args.env, exp_id)
if args.log:
    log_path = 'logs/log_{}_{}_{}.log'.format(args.model, args.env, exp_id)
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
else:
    logger = None
utils.logout(logger, 'Experiment: {}, Model: {}, Environment: {}, Seed: {}'.format(exp_id, args.model, repr(simulator),
                                                                                   args.seed))
utils.logout(logger,
             'gamma: {}, latent_dim: {}, lr: {}, batch_size: {}, eps_decay: {}, max steps: {}, latent_policy: {}, '
             'obs_normal: {}'.format(args.gamma, args.latent_dim, args.lr, args.batch_size, args.eps_decay,
                                     args.max_steps, args.latent_policy, args.obs_normal))
utils.logout(logger, 'CUDA is available: {}'.format(torch.cuda.is_available()))
utils.logout(logger, '*' * 50)

oderl = MBRL(simulator,
             gamma=args.gamma,
             mem_size=args.mem_size,
             latent_dim=args.latent_dim,
             batch_size=args.batch_size,
             lr=args.lr,
             ode_tol=args.ode_tol,
             ode_dim=args.ode_dim,
             enc_hidden_to_latent_dim=args.enc_hidden_to_latent_dim,
             eps_decay=args.eps_decay,
             model=args.model,
             timer_type=args.timer,
             latent_policy=args.latent_policy,
             obs_normal=args.obs_normal,
             exp_id=exp_id,
             trained_model_path=args.trained_model_path,
             ckpt_path=ckpt_path,
             logger=logger)

if args.train_env_model:
    utils.logout(logger, '*' * 10 + ' Collecting random rollouts ' + '*' * 10)
    for _ in range(args.trajs):
        oderl.run_policy(eps=1, max_steps=args.max_steps, store_trans=False, store_traj=True, optimize_mf=False,
                         cut_length=0, val_ratio=0)
    for _ in range(args.trajs // 10):
        oderl.run_policy(eps=1, max_steps=args.max_steps, store_trans=False, store_traj=True, optimize_mf=False,
                         cut_length=0, val_ratio=1)
    oderl.train_env_model(num_iters=args.iters)

if args.world_model:
    is_model_free = bool(args.model == 'free')
    choice = {True: oderl.run_policy, False: oderl.generate_traj_from_env_model}
    dic = {'rewards': [], 'trials': []}
    for i in range(args.episodes):
        t = time.time()
        choice[is_model_free](max_steps=args.max_steps)
        reward, _ = oderl.run_policy(eps=0.05, max_steps=args.max_steps, store_trans=False, store_traj=False,
                                     optimize_mf=False)
        dic['rewards'].append(reward)
        utils.logout(logger,
                     "Episode %d | rewards = %.6f | time = %.6f s" % (i + 1, dic['rewards'][-1], time.time() - t))
        if (i + 1) % 100 == 0:
            torch.save(dic, 'results_t/{}_{}_reward_{}.ckpt'.format(args.model, args.env, args.num_restarts))
    for _ in range(100):
        dic['trials'].append(oderl.run_policy(eps=0.05, max_steps=args.max_steps, store_trans=False, store_traj=False,
                                              optimize_mf=False)[0])
    utils.logout(logger, 'Average reward over last 100 trials: %f' % (sum(dic['trials'][-100:]) / 100))
    torch.save(dic, 'results/{}_{}_reward_{}.ckpt'.format(args.model, args.env, args.num_restarts))
    utils.logout(logger, '*' * 10 + ' Done ' + '*' * 10)

if args.mpc_ac:
    dic = {'rewards': [], 'trials': [], 'env_steps': []}
    total_env_steps = 0
    total_episodes = 0

    # random rollout
    rewards, steps, total_episodes, total_env_steps, eval_reward = \
        oderl.mbmf_rollout('random', 3 * args.env_steps, args.max_steps, total_episodes, total_env_steps, cur_epoch=0,
                           store_trans=True, store_traj=True, val_ratio=0.1, planning_horizon=args.planning_horizon)
    dic['env_steps'].extend(steps)
    dic['rewards'].extend(rewards)

    for i in range(max(args.mf_epochs, args.mb_epochs)):
        if i < args.mb_epochs:
            # model training
            utils.logout(logger, '*' * 10 + ' Training the environment model ' + '*' * 10)
            oderl.train_env_model_early_stopping(num_epochs=args.epochs, passes=max(15 - i, 3))

            # MBMF rollout
            rewards, steps, total_episodes, total_env_steps, eval_reward = \
                oderl.mbmf_rollout('mbmf', args.env_steps, args.max_steps, total_episodes, total_env_steps,
                                   cur_epoch=i + 1, store_trans=True, store_traj=True, val_ratio=0.1,
                                   planning_horizon=args.planning_horizon)
            dic['env_steps'].extend(steps)
            dic['rewards'].extend(rewards)
            dic['trials'].append(eval_reward)

        # MF rollout (only used for model-free policy)
        if i < args.mf_epochs:
            rewards, steps, total_episodes, total_env_steps, eval_reward = \
                oderl.mbmf_rollout('mf', args.env_steps, args.max_steps, total_episodes, total_env_steps,
                                   cur_epoch=i + 1, store_trans=True, store_traj=True, val_ratio=0.1,
                                   planning_horizon=args.planning_horizon)
            dic['env_steps'].extend(steps)
            dic['rewards'].extend(rewards)
            dic['trials'].append(eval_reward)
        torch.save(dic, 'results/{}_{}_reward_{}.ckpt'.format(args.model, args.env, args.num_restarts))
    utils.logout(logger, '*' * 10 + ' Done ' + '*' * 10)
