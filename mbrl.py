import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import torch.optim as optim
from policy import PolicyDQN, PolicyDDPG
from model import Encoder_z0_RNN, Decoder, DiffeqSolver, ODEFunc, Timer, MLPTimer, \
    VanillaGRU, DeltaTGRU, ExpDecayGRU, ODEGRU, VAEGRU, LatentODE
from replay_memory import ReplayMemory, PrioritizedReplayMemory, Transition, Trajectory
from running_stats import RunningStats
import utils


class MBRL(object):

    def __init__(self, simulator, gamma=0.99, mem_size=int(1e5), lr=9e-4, batch_size=32, ode_tol=1e-3, ode_dim=20,
                 enc_hidden_to_latent_dim=20, latent_dim=10, eps_decay=1e-4, weight_decay=1e-3, model=None, timer_type='',
                 latent_policy=False, obs_normal=False, exp_id=0, trained_model_path='', ckpt_path='',
                 traj_data_path='', logger=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_id = exp_id
        self.simulator = simulator
        self.batch_size = batch_size
        self.memory_traj_train = ReplayMemory(mem_size, Trajectory)
        self.memory_traj_test = ReplayMemory(mem_size // 10, Trajectory)
        self.input_dim = self.simulator.num_states + self.simulator.num_actions
        self.output_dim = self.simulator.num_states
        self.latent_dim = latent_dim
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.rms = RunningStats(dim=self.simulator.num_states, device=self.device) if obs_normal else None

        # policy and replay buffer
        assert not (model == 'free' and latent_policy)
        if 'HalfCheetah' in repr(simulator) or 'Swimmer' in repr(simulator) or 'Hopper' in repr(simulator):
            self.policy = PolicyDDPG(state_dim=self.simulator.num_states, action_dim=self.simulator.num_actions,
                                     device=self.device, gamma=gamma, latent=latent_policy)
            self.memory_trans = ReplayMemory(mem_size, Transition)
        else:
            state_dim = self.simulator.num_states + latent_dim if latent_policy else self.simulator.num_states
            self.policy = PolicyDQN(state_dim=state_dim, action_dim=self.simulator.num_actions,
                                    device=self.device, gamma=gamma, latent=latent_policy)
            self.memory_trans = PrioritizedReplayMemory(mem_size, Transition)

        # model
        min_t, max_t, max_time_length, is_cont = simulator.get_time_info()
        timer_choice = Timer if timer_type == 'fool' else MLPTimer
        timer = timer_choice(input_dim=self.input_dim + self.latent_dim,
                             output_dim=1 if is_cont else max_t - min_t + 1,
                             min_t=min_t, max_t=max_t, max_time_length=max_time_length,
                             device=self.device).to(self.device)

        # ode network
        if 'ode' in model:
            gen_ode_func = ODEFunc(ode_func_net=utils.create_net(latent_dim, latent_dim, n_layers=2, n_units=ode_dim,
                                                                 nonlinear=nn.Tanh)).to(self.device)
            diffq_solver = DiffeqSolver(gen_ode_func, 'dopri5', odeint_rtol=ode_tol, odeint_atol=ode_tol/10)

        # encoder
        if model == 'vae-rnn' or model == 'latent-ode':
            encoder = Encoder_z0_RNN(latent_dim, self.input_dim, hidden_to_z0_units=enc_hidden_to_latent_dim,
                                     device=self.device).to(self.device)
            z0_prior = Normal(torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device))

        # decoder
        decoder = Decoder(latent_dim, self.output_dim, n_layers=0).to(self.device)

        if model == 'free' or model == 'rnn':
            self.model = VanillaGRU(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                decoder=decoder,
                timer=timer,
                device=self.device).to(self.device)
        elif model == 'deltaT-rnn':
            self.model = DeltaTGRU(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                decoder=decoder,
                timer=timer,
                device=self.device).to(self.device)
        elif model == 'decay-rnn':
            self.model = ExpDecayGRU(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                decoder=decoder,
                timer=timer,
                device=self.device).to(self.device)
        elif model == 'ode-rnn':
            self.model = ODEGRU(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                decoder=decoder,
                diffeq_solver=diffq_solver,
                timer=timer,
                device=self.device).to(self.device)
        elif model == 'vae-rnn':
            self.model = VAEGRU(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=decoder,
                z0_prior=z0_prior,
                timer=timer,
                device=self.device).to(self.device)
        elif model == 'latent-ode':
            self.model = LatentODE(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=decoder,
                diffeq_solver=diffq_solver,
                z0_prior=z0_prior,
                timer=timer,
                device=self.device).to(self.device)
        else:
            raise NotImplementedError

        if trained_model_path:
            self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device)['model_state_dict'])

        if traj_data_path:
            self.load_traj_buffer(traj_data_path)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_env_model(self, num_iters=12000, log_every=200):
        """
            Train environment model with replay buffer
        """
        if len(self.memory_traj_train) < self.batch_size:
            return

        train_mses, test_mses, test_mses_by_state = [], [], []
        t = time.time()
        for i in range(num_iters):
            trajs = self.memory_traj_train.sample(self.batch_size)
            batch = Trajectory(*zip(*trajs))
            lengths_batch = torch.tensor(batch.length, dtype=torch.long, device=self.device)  # [N,]
            max_length = lengths_batch.max()
            states_batch = torch.stack(batch.states)[:, :max_length + 1, :]  # [N, T+1, D_state]
            actions_batch = torch.stack(batch.actions)[:, :max_length, :]  # [N, T, D_action]
            time_steps_batch = torch.stack(batch.time_steps)[:, :max_length + 1]  # [N, T+1]

            # compute loss
            train_losses_dict = self.model.compute_loss(states_batch, actions_batch, time_steps_batch, lengths_batch,
                                                        train=True)
            train_total_loss = train_losses_dict['total']
            train_mses.append(train_losses_dict['mse'].item())

            # optimize
            self.optimizer.zero_grad()
            train_total_loss.backward()
            self.optimizer.step()

            if (i + 1) % log_every == 0:
                if len(self.memory_traj_test) >= self.batch_size:
                    trajs_test = self.memory_traj_test.sample(self.batch_size)
                    batch_test = Trajectory(*zip(*trajs_test))
                    lengths_batch_test = torch.tensor(batch_test.length, dtype=torch.long, device=self.device)
                    max_length_test = lengths_batch_test.max()
                    states_batch_test = torch.stack(batch_test.states)[:, :max_length_test + 1, :]
                    actions_batch_test = torch.stack(batch_test.actions)[:, :max_length_test, :]
                    time_steps_batch_test = torch.stack(batch_test.time_steps)[:, :max_length_test + 1]
                    with torch.no_grad():
                        test_losses_dict = self.model.compute_loss(states_batch_test, actions_batch_test,
                                                                   time_steps_batch_test, lengths_batch_test,
                                                                   train=False)
                        test_mses.append(test_losses_dict['mse'].item())
                    log = "Iter %d | training MSE = %.6f | test MSE = %.6f | " \
                          "training dt loss = %.6f | test dt loss = %.6f" % (i + 1, train_mses[-1], test_mses[-1],
                                                                             train_losses_dict['dt'].item(),
                                                                             test_losses_dict['dt'].item())
                else:
                    log = "Iter %d | training MSE = %.6f | training dt loss = %.6f" % \
                          (i + 1, train_mses[-1], train_losses_dict['dt'].item())
                if 'kl' in train_losses_dict:
                    log += " | training KL = %.6f" % (train_losses_dict['kl'].item())  # no test KL
                log += " | time = %.6f s" % (time.time() - t)
                if self.logger:
                    self.logger.info(log)
                else:
                    print(log)

                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_train_loss': train_mses,
                    'model_test_loss': test_mses,
                    'model_test_loss_by_state': test_mses_by_state,
                }, self.ckpt_path)
                t = time.time()

    def train_env_model_early_stopping(self, num_epochs=200, passes=20):
        """
            Train environment model with replay buffer
        """
        best_loss, best_loss_idx, model_dict = None, None, None
        train_dataset = TensorDataset(*self.read_trajs_from_buffer(self.memory_traj_train, len(self.memory_traj_train)))
        test_dataset = TensorDataset(*self.read_trajs_from_buffer(self.memory_traj_test, len(self.memory_traj_test)))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_mses = []
        t = time.time()

        def run_epoch(loader, train=True):
            if train:
                self.model.train()
            else:
                self.model.eval()
            total_mse_loss, total_dt_loss = 0, 0
            num_iters = 0
            for i, (states, actions, time_steps, lengths) in enumerate(loader):
                max_length = lengths.max()
                states = states[:, :max_length + 1, :].to(self.device)  # [B, T+1, D_state]
                actions = actions[:, :max_length, :].to(self.device)  # [B, T, D_action]
                time_steps = time_steps[:, :max_length + 1].to(self.device)  # [B, T+1]
                losses_dict = self.model.compute_loss(states, actions, time_steps, lengths.to(self.device), train=train)
                loss = losses_dict['total']
                total_mse_loss += losses_dict['mse'].item()
                total_dt_loss += losses_dict['dt'].item()
                num_iters += 1

                # optimize
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            return {'mse': total_mse_loss / num_iters, 'dt': total_dt_loss / num_iters}, num_iters

        for e in range(num_epochs):
            train_loss_dict, num_iters = run_epoch(train_loader, train=True)
            with torch.no_grad():
                test_loss_dict, _ = run_epoch(test_loader, train=False)
            test_mses.append(test_loss_dict['mse'])
            if best_loss is None or test_mses[-1] < best_loss:
                best_loss = test_mses[-1]
                best_loss_idx = e
                model_dict = self.model.state_dict()
                torch.save({
                    'model_state_dict': model_dict,
                    'model_test_loss': test_mses,
                    'model_best_test_loss': best_loss
                }, self.ckpt_path)
            log = "Epoch %d | training MSE = %.6f | test MSE = %.6f | training dt loss = %.6f |" \
                  " test dt loss = %.6f" % (e + 1, train_loss_dict['mse'], test_loss_dict['mse'],
                                            train_loss_dict['dt'], test_loss_dict['dt'])
            if 'kl' in train_loss_dict:
                log += " | training kl = %.6f" % train_loss_dict['kl']
            log += " | time = %.6f s" % (time.time() - t)
            utils.logout(self.logger, log)
            t = time.time()

            if e - best_loss_idx >= passes:
                break

        utils.logout(self.logger, 'Finish training model, best test MSE: %.6f' % best_loss)
        self.model.load_state_dict(model_dict)

    def run_policy(self, max_steps, eps=None, store_trans=True, optimize_mf=True, store_traj=False, val_ratio=0,
                   cut_length=0):
        """
            Run policy once with interaction with environment, optimize policy and save transitions and trajectories
            Note that when we call this function, env model should been trained to generate reasonable latent states
        """
        latent_states = [self.model.sample_init_latent_states() if self.policy.latent else None]
        states = [torch.tensor(self.simulator.reset(), dtype=torch.float, device=self.device)]
        actions_encoded, rewards, dts = [], [], [0.]
        length = 0
        for _ in range(max_steps):
            state = states[-1]
            latent_state = latent_states[-1]
            if self.rms is not None:
                self.rms += state
            norm_state = state if self.rms is None else self.rms.normalize(state)
            action = self.policy.select_action(norm_state if not self.policy.latent or latent_state is None
                                               else torch.cat((norm_state, latent_state)), eps=eps)
            action_encoded = self.policy.func_encode_action([action], self.simulator.num_actions, self.device)
            dt = self.simulator.get_time_gap(action=action)
            next_state, reward, done, info = self.simulator.step(action, dt=dt)
            if self.policy.latent and (eps is None or eps < 1):
                state_action = torch.cat((norm_state.unsqueeze(0), action_encoded), dim=-1)
                next_latent_state = self.model.encode_next_latent_state(state_action, latent_state.unsqueeze(0),
                                                                        torch.tensor([dt], dtype=torch.float,
                                                                                     device=self.device)).detach().squeeze(0)
            else:
                next_latent_state = None
            next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
            states.append(next_state)
            latent_states.append(next_latent_state)
            actions_encoded.append(action_encoded)
            rewards.append(reward)
            dts.append(dt)
            length += 1

            # store to trans buffer
            if store_trans:
                self.save_trans_to_buffer(state, next_state, action, reward, latent_state, next_latent_state, dt, done)

            # optimize if not eval mode
            if optimize_mf:
                self.policy.optimize(self.memory_trans, Transition, self.rms)

            if done:
                break
        if optimize_mf:
            self.policy.update_target_policy()

        # calculate accumulated rewards
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)  # [T,]
        time_steps = torch.tensor(dts, device=self.device, dtype=torch.float).cumsum(dim=0)  # [T+1, ]
        acc_rewards = self.calc_acc_rewards(rewards, time_steps[:-1],
                                            discount=bool('HIV' in repr(self.simulator))).item()

        # store to traj buffer
        if store_traj:
            states = torch.stack(states)  # [T+1, D_state]
            actions_encoded = torch.stack(actions_encoded).squeeze(1)  # [T, D_action]
            states, actions_encoded, time_steps = self.pad_trajectories(states, actions_encoded, time_steps, length,
                                                                       cut_length, max_steps)
            self.save_traj_to_buffer(states, actions_encoded, time_steps, length, cut_length, val_ratio)

        return acc_rewards, length

    def generate_traj_from_env_model(self, max_steps, store_trans=True, optimize_mf=True, env_init=True):
        """
            Simulate trajectories using learned model instead of interaction with environment
        """
        self.model.timer.reset()  # reset model's timer every time
        latent_state = self.model.sample_init_latent_states()
        if env_init or len(self.memory_traj_train) == 0:
            state = torch.tensor(self.simulator.reset(), dtype=torch.float, device=self.device)
        else:
            self.simulator.reset()
            trajs = self.memory_traj_train.sample(1)
            batch = Trajectory(*zip(*trajs))
            state = batch.states[0][np.random.randint(0, batch.length[0]+1)].to(self.device)
        if self.rms is not None:
            state = self.rms.normalize(state)

        # simulate
        for _ in range(max_steps):
            # select action based on state
            action = self.policy.select_action(state if not self.policy.latent else torch.cat((state, latent_state)))
            action_encoded = self.policy.func_encode_action([action], self.policy.num_actions, self.device)
            dt = self.model.timer.deliver_dt(torch.cat((state, action_encoded.squeeze(0), latent_state)))
            state_action = torch.cat((state.unsqueeze(0), action_encoded), dim=-1)
            next_latent_state = self.model.encode_next_latent_state(state_action, latent_state.unsqueeze(0),
                                                                    torch.tensor([dt], dtype=torch.float,
                                                                                 device=self.device)
                                                                    ).detach().squeeze(0)
            next_state = self.model.decode_latent_traj(next_latent_state).detach()
            reward = self.simulator.calc_reward(action=action, state=next_state.cpu().numpy(), dt=dt)
            done = self.simulator.is_terminal(state=next_state.cpu().numpy()) or self.model.timer.is_terminal()

            # store to replay buffer
            if store_trans:
                self.save_trans_to_buffer(state, next_state, action, reward, latent_state, next_latent_state, dt, done)

            if optimize_mf:
                self.policy.optimize(self.memory_trans, Transition, self.rms)

            if done:
                break
            state = next_state
            latent_state = next_latent_state
        if optimize_mf:
            self.policy.update_target_policy()

    def mpc_planning(self, max_steps, planning_horizon=20, search_population=1000, store_trans=True, store_traj=False,
                     val_ratio=0, cut_length=0, rand=False, combine_mf=False, soft_num=50):
        latent_states = [self.model.sample_init_latent_states()]
        states = [torch.tensor(self.simulator.reset(), dtype=torch.float, device=self.device)]
        actions_encoded, rewards, dts = [], [], [0.]
        length = 0
        for _ in range(max_steps):
            state = states[-1]
            latent_state = latent_states[-1]
            if self.rms is not None:
                self.rms += state
            norm_state = state if self.rms is None else self.rms.normalize(state)
            if rand:
                action = np.random.uniform(-1, 1, size=self.simulator.num_actions)
            else:
                action = self.mpc_search(norm_state, latent_state, planning_horizon, search_population, soft_num,
                                         combine_mf=combine_mf)
            action_encoded = torch.tensor([action], dtype=torch.float, device=self.device)
            dt = self.simulator.get_time_gap(action=action)
            next_state, reward, done, info = self.simulator.step(action, dt=dt)
            state_action = torch.cat((norm_state.unsqueeze(0), action_encoded), dim=-1)
            if not rand:
                next_latent_state = self.model.encode_next_latent_state(state_action, latent_state.unsqueeze(0),
                                                                        torch.tensor([dt], dtype=torch.float,
                                                                                     device=self.device)).detach().squeeze(0)
            else:
                next_latent_state = None
            next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
            states.append(next_state)
            latent_states.append(next_latent_state)
            actions_encoded.append(action_encoded)
            rewards.append(reward)
            dts.append(dt)
            length += 1

            # store to trans buffer
            if store_trans:
                self.save_trans_to_buffer(state, next_state, action, reward, latent_state, next_latent_state, dt, done)

            if combine_mf:
                self.policy.optimize(self.memory_trans, Transition, self.rms)

            if done:
                break

        # calculate accumulated rewards
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)  # [T,]
        time_steps = torch.tensor(dts, device=self.device, dtype=torch.float).cumsum(dim=0)  # [T+1, ]
        acc_rewards = self.calc_acc_rewards(rewards, time_steps[:-1]).item()

        # store to traj buffer
        if store_traj:
            states = torch.stack(states)  # [T+1, D_state]
            actions_encoded = torch.stack(actions_encoded).squeeze(1)  # [T, D_action]
            states, actions_encoded, time_steps = self.pad_trajectories(states, actions_encoded, time_steps, length,
                                                                        cut_length, max_steps)
            self.save_traj_to_buffer(states, actions_encoded, time_steps, length, cut_length, val_ratio)

        return acc_rewards, length

    def mpc_search(self, state, latent_state, h, k, e=50, combine_mf=False):
        actions_list = torch.empty(k, h, self.simulator.num_actions, dtype=torch.float,
                                   device=self.device)  # [K, H, D_action]
        states_list = torch.empty(k, h+1, self.simulator.num_states, dtype=torch.float,
                                  device=self.device)   # [K, H+1, D_state]
        dts_list = torch.empty(k, h, dtype=torch.float, device=self.device)  # [K, H]
        states_list[:, 0, :] = state.repeat(k, 1)
        latent_states = latent_state.repeat(k, 1)  # [K, D_latent]
        with torch.no_grad():
            for i in range(h):
                if combine_mf:
                    actions_list[:, i, :] = self.policy.select_action_in_batch(states_list[:, i, :])
                else:
                    actions_list[:, i, :].uniform_(-1, 1)
                data = torch.cat((states_list[:, i, :], actions_list[:, i, :]), dim=-1)  # [K, D_state+D_action]
                dts_list[:, i] = self.model.timer.deliver_dt_in_batch(torch.cat((data, latent_states), dim=-1))
                next_latent_states = self.model.encode_next_latent_state(data, latent_states,
                                                                         dts_list[:, i])  # [K, D_latent]
                states_list[:, i+1, :] = self.model.decode_latent_traj(next_latent_states)  # [K, D_state]
                latent_states = next_latent_states
        rewards_list, masks = self.simulator.calc_reward_in_batch(states_list if self.rms is None else
                                                                  self.rms.unnormalize(states_list),
                                                                  actions_list, dts_list)  # [K, H]
        time_steps_list = dts_list.cumsum(dim=1)
        rewards = self.calc_acc_rewards(rewards_list, time_steps_list, discount=True)
        if combine_mf:
            last_actions = self.policy.select_action_in_batch(states_list[:, -1, :], noise=False)
            last_states_value = self.policy.calc_value_in_batch(states_list[:, -1, :], last_actions)
            rewards[masks[:, -1]] += (self.policy.gamma ** time_steps_list[:, -1] * last_states_value)[masks[:, -1]]
        best_actions = actions_list[torch.topk(rewards, e, sorted=False)[1], 0, :]  # soft greedy
        return best_actions.mean(dim=0).cpu().numpy()

    def model_rollout(self, states, h):
        states_list, actions_list, dts_list = [states], [], []
        cur_latent_states = self.model.sample_init_latent_states(num_trajs=states.size(0))  # [B, D_latent]
        with torch.no_grad():
            for i in range(h):
                cur_states = states_list[-1]  # [B, D_state]
                cur_actions = self.policy.select_action_in_batch(cur_states, target=True)  # [B, D_action]
                cur_states_actions = torch.cat((cur_states, cur_actions), dim=-1)  # [B, D_state+D_action]
                cur_dts = self.model.timer.deliver_dt_in_batch(torch.cat((cur_states_actions,
                                                                          cur_latent_states), dim=-1))  # [B,]
                next_latent_states = self.model.encode_next_latent_state(cur_states_actions, cur_latent_states,
                                                                         cur_dts)  # [B, D_latent]
                next_states = self.model.decode_latent_traj(next_latent_states)  # [B, D_state]
                states_list.append(next_states)
                actions_list.append(cur_actions)
                dts_list.append(cur_dts)
                cur_latent_states = next_latent_states
        states_list = torch.stack(states_list).permute(1, 0, 2)  # [B, H+1, D_action]
        actions_list = torch.stack(actions_list).permute(1, 0, 2)  # [B, H, D_action]
        dts_list = torch.stack(dts_list).permute(1, 0)  # [B, H]
        rewards_list, masks = self.simulator.calc_reward_in_batch(states_list if self.rms is None else
                                                                  self.rms.unnormalize(states_list),
                                                                  actions_list, dts_list)  # [K, H]
        return states_list, actions_list, dts_list, rewards_list, masks

    def calc_acc_rewards(self, rewards, time_steps, discount=False):
        """
            Calculate accumulated return base on semi-mdp
        """
        discounts = self.policy.gamma ** time_steps if discount \
            else torch.ones_like(time_steps, dtype=torch.float, device=self.device)
        if len(rewards.size()) == 1:  # [T,]
            return torch.dot(rewards, discounts)
        elif len(rewards.size()) == 2:  # [N, T]
            return torch.mm(rewards, discounts.t()).diag()
        else:
            raise ValueError("rewards should be 1D vector or 2D matrix.")

    def load_traj_buffer(self, path):
        data = torch.load(path, map_location=self.device)
        self.memory_traj_train = data['train']
        self.memory_traj_test = data['test']

    def save_traj_to_buffer(self, states, actions_encoded, time_steps, length, unit, val_ratio):
        idx = 0
        if unit > 0:
            T = actions_encoded.size(0)
            assert T != 0 and T % unit == 0
            while idx < length:
                memory = self.memory_traj_train if np.random.uniform() > val_ratio else self.memory_traj_test
                memory.push(states[idx:idx+unit+1], actions_encoded[idx:idx+unit],
                            time_steps[idx:idx+unit+1], unit if idx+unit < length else length-idx)
                idx += unit
        else:
            memory = self.memory_traj_train if np.random.uniform() > val_ratio else self.memory_traj_test
            memory.push(states, actions_encoded, time_steps, length)

    def read_trajs_from_buffer(self, memory, batch_size):
        trajs = memory.sample(batch_size)
        batch = Trajectory(*zip(*trajs))
        states_batch = torch.stack(batch.states)  # [N, T+1, D_state]
        if self.rms is not None:
            states_batch = self.rms.normalize(states_batch)
        actions_batch = torch.stack(batch.actions)  # [N, T, D_action]
        time_steps_batch = torch.stack(batch.time_steps)  # [N, T+1]
        lengths_batch = torch.tensor(batch.length)  # [N,]
        return states_batch, actions_batch, time_steps_batch, lengths_batch

    def save_trans_to_buffer(self, state, next_state, action, reward, latent_state, next_latent_state, dt, done):
        if done:
            next_state, next_latent_state = None, None
        if repr(self.policy) == 'DQN':
            error = self.policy.calc_td_error(state, next_state, action, reward, latent_state, next_latent_state, dt)
            self.memory_trans.push(state, next_state, action, reward, latent_state, next_latent_state, dt, priority=error)
        else:
            self.memory_trans.push(state, next_state, action, reward, latent_state, next_latent_state, dt)

    def pad_trajectories(self, states, actions, time_steps, length, cut_length, max_steps):
        """
            Pad the trajectory to fixed max_steps for mini-batch learning
        """
        if cut_length == 0:
            pad_length = max_steps - length
        else:
            pad_length = ((length-1) // cut_length + 1) * cut_length - length
        if pad_length > 0:
            states = torch.cat((states, torch.zeros(pad_length, self.simulator.num_states, dtype=torch.float,
                                                    device=self.device)))
            actions = torch.cat((actions, torch.zeros(pad_length, self.simulator.num_actions, dtype=torch.float,
                                                      device=self.device)))
            time_steps = torch.cat((time_steps, torch.full((pad_length,), time_steps[-1].item(), dtype=torch.float,
                                                           device=self.device)))
        return states, actions, time_steps

    def mbmf_rollout(self, mode, env_steps, max_steps, total_episodes, total_env_steps, cur_epoch, store_trans=True,
                     store_traj=False, planning_horizon=None, val_ratio=0):
        t = time.time()
        cur_steps = 0
        rewards, steps = [], []
        utils.logout(self.logger, '*'*10 + ' {} rollout '.format(mode.upper()) + '*'*10)
        while cur_steps < env_steps:
            actual_steps = min(max_steps, env_steps-cur_steps)
            if 'mb' in mode:
                reward, step = self.mpc_planning(actual_steps, store_trans=store_trans, store_traj=store_traj,
                                                 val_ratio=val_ratio, cut_length=planning_horizon, rand=False,
                                                 planning_horizon=planning_horizon, combine_mf=bool(mode == 'mbmf'))
            elif mode == 'mf':
                reward, step = self.run_policy(actual_steps, store_trans=store_trans, store_traj=store_traj,
                                               optimize_mf=True, val_ratio=val_ratio, cut_length=planning_horizon)
            elif mode == 'random':
                reward, step = self.run_policy(actual_steps, store_trans=store_trans, store_traj=store_traj,
                                               optimize_mf=False, eps=1, val_ratio=val_ratio, cut_length=planning_horizon)
            else:
                raise ValueError('only MBMF, MB, MF or Random.')
            cur_steps += step
            total_env_steps += step
            total_episodes += 1
            rewards.append(reward)
            steps.append(total_env_steps)
            log = "Episode {} | total env steps = {} | env steps = {} | reward = {:.6f}".format(total_episodes,
                                                                                                total_env_steps,
                                                                                                step, reward)
            utils.logout(self.logger, log)

        utils.logout(self.logger, '*'*10 + ' Policy evaluation ' + '*'*10)
        eval_reward = 0
        for _ in range(5):
            eval_reward += self.run_policy(max_steps=max_steps, store_trans=False, store_traj=False,
                                           optimize_mf=False)[0]
        log = "{} Epoch {} | total env steps = {} | avg reward over last epoch = {:.6f} | eval reward = {:.6f}" \
              " | time = {:.6f} s".format(mode.upper(), cur_epoch, total_env_steps, sum(rewards) / len(rewards),
                                          eval_reward / 5, time.time() - t)
        utils.logout(self.logger, log)

        return rewards, steps, total_episodes, total_env_steps, eval_reward

    def select_best_tau(self, action, state_action, latent_state, gamma=None, oracle=False):
        min_t, max_t, _, is_cont = self.simulator.get_time_info()
        assert not is_cont
        time_steps = torch.arange(max_t+1, dtype=torch.float, device=self.device)
        rewards = []
        if gamma is None:
            gamma = self.policy.gamma
        if oracle:
            from copy import deepcopy
            next_states = []
            simulator = deepcopy(self.simulator)
            for _ in range(min_t, max_t+1):
                next_state, reward, done, info = simulator.step(action, dt=1)
                next_states.append(next_state)
                rewards.append(reward)
            next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        else:
            with torch.no_grad():
                traj_latent_states = self.model.rollout_timeline(state_action, latent_state.unsqueeze(0),
                                                                 time_steps).squeeze(0)
                next_states = self.model.decode_latent_traj(traj_latent_states[1:, :])  # [T, D]
            for next_state in next_states:
                rewards.append(self.simulator.calc_reward(action=action, state=next_state.cpu().numpy(), dt=1))
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_actions = self.policy.select_action_in_batch(next_states)
        values = self.policy.calc_value_in_batch(next_states, next_actions.long())
        discount_values = values * (gamma ** time_steps[1:])
        discount_rewards = torch.cumsum(rewards * (gamma ** time_steps[:-1]), dim=0)
        acc_rewards = discount_rewards + discount_values
        best_tau = acc_rewards.argmax().item() + 1
        return best_tau, traj_latent_states[best_tau] if not oracle else None, next_states[best_tau - 1], \
               discount_rewards[best_tau - 1]
