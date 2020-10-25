import random
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.nn.modules.rnn import GRU, GRUCell
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions.normal import Normal
from torchdiffeq import odeint as odeint
import utils


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net, nonlinear=None):
        super(ODEFunc, self).__init__()
        self.net = ode_func_net
        self.nonlinear = nonlinear

    def forward(self, t, x):
        """
        Perform one step in solving ODE.
        """
        return self.nonlinear(self.net(x)) if self.nonlinear else self.net(x)


class DiffeqSolver(nn.Module):

    def __init__(self, ode_func, method, odeint_rtol, odeint_atol):
        super(DiffeqSolver, self).__init__()
        self.ode_func = ode_func
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps, odeint_rtol=None, odeint_atol=None, method=None):
        """
            Decode the trajectory through ODE Solver
            @:param first_point, shape [N, D]
                    time_steps, shape [T,]
            @:return predicted the trajectory, shape [N, T, D]
        """
        if not odeint_rtol:
            odeint_rtol = self.odeint_rtol
        if not odeint_atol:
            odeint_atol = self.odeint_atol
        if not method:
            method = self.ode_method
        pred = odeint(self.ode_func, first_point, time_steps,
                      rtol=odeint_rtol, atol=odeint_atol, method=method)  # [T, N, D]
        pred = pred.permute(1, 0, 2)  # [N, T, D]
        assert (torch.mean(pred[:, 0, :] - first_point) < 0.001)  # the first prediction is same with first point
        assert pred.size(0) == first_point.size(0)
        assert pred.size(1) == time_steps.size(0)
        assert pred.size(2) == first_point.size(1)
        return pred


class Encoder_z0_RNN(nn.Module):

    def __init__(self, latent_dim, input_dim, device, hidden_to_z0_units=20, bidirectional=False):
        super(Encoder_z0_RNN, self).__init__()
        self.device = device
        self.latent_dim = latent_dim  # latent dim for z0 and encoder rnn
        self.input_dim = input_dim
        self.hidden_to_z0 = nn.Sequential(
            nn.Linear(2 * latent_dim if bidirectional else latent_dim, hidden_to_z0_units),
            nn.Tanh(),
            nn.Linear(hidden_to_z0_units, 2 * latent_dim))
        self.rnn = GRU(input_dim, latent_dim, batch_first=True, bidirectional=bidirectional).to(device)

    def forward(self, data, time_steps, lengths):
        """
            Encode the mean and log variance of initial latent state z0
            @:param data, shape [N, T, D]
                    time_steps, shape [N, T]
                    lengths, shape [N,]
            @:return mean, logvar of z0, shape [N, D_latent]
        """
        data_packed = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(data_packed)
        assert hidden.size(1) == data.size(0)
        assert hidden.size(2) == self.latent_dim

        # check if bidirectional
        if hidden.size(0) == 1:
            hidden = hidden.squeeze(0)
        elif hidden.size(0) == 2:
            hidden = torch.cat((hidden[0], hidden[1]), dim=-1)
        else:
            raise ValueError('Incorrect RNN hidden state.')

        # extract mean and logvar
        mean_logvar = self.hidden_to_z0(hidden)
        assert mean_logvar.size(-1) == 2 * self.latent_dim
        mean, logvar = mean_logvar[:, :self.latent_dim], mean_logvar[:, self.latent_dim:]
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, n_layers=0, n_units=0):
        super(Decoder, self).__init__()
        self.decoder = utils.create_net(latent_dim, input_dim, n_layers=n_layers, n_units=n_units, nonlinear=nn.ReLU)

    def forward(self, data):
        return self.decoder(data)


class Timer(nn.Module):
    """
        Timer without learning to timing
    """

    def __init__(self, input_dim, output_dim, min_t, max_t, max_time_length, device):
        super(Timer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim > 1:  # discrete
            assert max_t - min_t + 1 == output_dim
            self.is_continuous = False
        else:
            self.is_continuous = True
        self.min_t = min_t
        self.max_t = max_t
        self.max_time_length = max_time_length
        self.t = 0

    def __repr__(self):
        return 'Timer'

    def compute_loss(self, x, dts, masks):
        raise NotImplementedError

    def deliver_dt(self, x, choice='random'):
        if choice == 'min':
            dt = self.min_t
        elif choice == 'max':
            dt = self.max_t
        elif choice == 'mean':
            t_mean = (self.min_t + self.max_t) / 2
            dt = t_mean if self.is_continuous else int(t_mean)
        elif choice == 'random':
            dt = random.uniform(self.min_t, self.max_t) if self.is_continuous \
                else random.randint(self.min_t, self.max_t)
        elif (type(choice) == int or type(choice) == float) and self.min_t <= choice <= self.max_t:
            dt = choice
        else:
            raise NotImplementedError
        self.t += dt
        return dt

    def get_time_info(self):
        return self.min_t, self.max_t, self.is_continuous

    def is_terminal(self):
        return self.t >= self.max_time_length

    def reset(self):
        self.t = 0


class MLPTimer(Timer):
    """
        Timer with learning to timing using MLP
    """

    def __init__(self, input_dim, output_dim, min_t, max_t, max_time_length, device):
        super(MLPTimer, self).__init__(input_dim, output_dim, min_t, max_t, max_time_length, device)
        self.net = utils.create_net(input_dim, output_dim, n_layers=1, n_units=20, nonlinear=nn.Tanh)
        self.criterion = nn.MSELoss() if self.is_continuous else nn.CrossEntropyLoss()

    def __repr__(self):
        return 'MLPTimer'

    def compute_loss(self, x, dts, masks):
        """
            Compute MSE or CE loss for learning time interval
            @:param x, input [N, T, D]
                    dts, target [N, T]
                    lengths, shape [N,]
            @:return loss
        """
        pred_dts = self.net(x)
        assert pred_dts.size(0) == dts.size(0)
        assert pred_dts.size(1) == dts.size(1)
        if self.is_continuous:
            return self.criterion(pred_dts[masks].squeeze(-1), dts[masks])
        else:
            return self.criterion(pred_dts[masks], dts[masks].long() - self.min_t)

    def deliver_dt(self, x, choice='learned'):
        """
            Generate one-step time gap given current state and action
            @:param x, input, [D,]
            @:return: dt
        """
        if choice == 'learned':
            with torch.no_grad():
                x = self.net(x)
                dt = self.min_t + x.argmax(dim=-1).item() if not self.is_continuous \
                    else torch.clamp(x, self.min_t, self.max_t).item()
                self.t += dt
        else:
            dt = super().deliver_dt(x, choice)
        return dt

    def deliver_dt_in_batch(self, x):
        """
            Generate a batch of one-step time gap given current states and actions
            @:param x, input, [N, D]
            @:return: dt, [N,]
        """
        with torch.no_grad():
            x = self.net(x)
            dts = self.min_t + x.argmax(dim=-1) if not self.is_continuous \
                else torch.clamp(x, self.min_t, self.max_t).squeeze(-1)
        return dts.float()


class BaseRecurrentModel(nn.Module):
    """
        Base recurrent model as an abstract class
    """

    def __init__(self, input_dim, latent_dim, eps_decay, decoder, timer, device):
        super(BaseRecurrentModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.decoder = decoder
        self.timer = timer
        self.eps = 1.
        self.eps_decay = eps_decay
        self.i_step = 0
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "BaseRecurrentModel"

    def decay_eps(self):
        """
            Linear decay
        """
        if self.eps_decay > 0 and self.eps > 0:
            self.eps = max(0, 1. - self.eps_decay * self.i_step)
        self.i_step += 1

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        raise NotImplementedError("Abstract class cannot be used.")

    def encode_latent_traj(self, states, actions, time_steps, train=True):
        """
            Encode latent trajectories given states, actions and timesteps
            @:param states, shape [N, T, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T+1]
            @:return hs, shape [N, T+1, D_latent]
        """
        N = states.size(0)
        pred_next_states = []
        hs = [self.sample_init_latent_states(num_trajs=N)]  # hs[-1] [N, D_latent]
        for i in range(time_steps.size(1) - 1):
            if i == 0 or (train and self.eps_decay == 0):
                data = torch.cat((states[:, i, :], actions[:, i, :]), dim=-1)  # [N, D_state+D_action]
            else:
                data = torch.cat((pred_next_states[-1], actions[:, i, :]), dim=-1)
                if train and self.eps > 0:  # scheduled sampling
                    heads = torch.rand(N) < self.eps  # [N,]
                    data[heads] = torch.cat((states[:, i, :], actions[:, i, :]), dim=-1)[heads]
            hs.append(self.encode_next_latent_state(data, hs[-1], time_steps[:, i + 1] - time_steps[:, i]))
            pred_next_states.append(self.decode_latent_traj(hs[-1]))
        hs = torch.stack(hs).permute(1, 0, 2)  # [N, T+1, D_latent]
        pred_next_states = torch.stack(pred_next_states).permute(1, 0, 2)  # [N, T, D_state]
        if train:
            self.decay_eps()

        assert hs.size(0) == N
        assert hs.size(1) == time_steps.size(1)
        assert hs.size(2) == self.latent_dim
        return hs, pred_next_states

    def decode_latent_traj(self, hs):
        """
            Decode latent trajectories
            @:param hs, shape [N, T, D_latent]
            @:return shape [N, T, D_state]
        """
        return self.decoder(hs)

    def predict_next_states(self, states, actions, time_steps, train=True):
        """
            Predict next states given states, actions and timesteps
            @:param states, shape [N, T, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T+1]
            @:return next_states, shape [N, T, D_state]
                     hs (current latent states), shape [N, T, D_latent]
        """
        # encoding and decoding
        hs, next_states = self.encode_latent_traj(states, actions, time_steps, train=train)  # [N, T+1, D_latent]
        return next_states, hs[:, :-1, :]

    def compute_loss(self, states, actions, time_steps, lengths, dt_coef=.01, train=True):
        """
            Compute RNN loss
            @:param states, shape [N, T+1, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T+1]
                    lengths, shape [N,]
                    dt_coef, the coefficient before dt loss term
            @:return total loss = mse + dt loss,
                     mse loss
        """
        # predict next states
        traj_cur_states, traj_next_states = states[:, :-1, :], states[:, 1:, :]
        traj_pred_next_states, traj_cur_latent_states = self.predict_next_states(traj_cur_states, actions, time_steps,
                                                                                 train=train)
        max_len = lengths.max()
        masks = torch.arange(max_len, device=self.device).expand(lengths.size(0), max_len) < lengths.unsqueeze(1)
        mse_loss = self.criterion(traj_next_states[masks], traj_pred_next_states[masks])

        # loss for time gap prediction
        if repr(self.timer) != 'Timer':
            dts = time_steps[:, 1:] - time_steps[:, :-1]
            dt_loss = self.timer.compute_loss(torch.cat((traj_cur_states, actions, traj_cur_latent_states), dim=-1),
                                              dts, masks)
        else:
            dt_loss = torch.tensor([0.], device=self.device)

        return {'total': mse_loss + dt_coef * dt_loss, 'mse': mse_loss, 'dt': dt_loss}

    def sample_init_latent_states(self, num_trajs=0):
        shape = (self.latent_dim,) if num_trajs == 0 else (num_trajs, self.latent_dim)
        return torch.zeros(shape, dtype=torch.float, device=self.device)


class VanillaGRU(BaseRecurrentModel):
    """
        Vanilla GRU
    """

    def __init__(self, input_dim, latent_dim, eps_decay, decoder, timer, device):
        super(VanillaGRU, self).__init__(input_dim, latent_dim, eps_decay, decoder, timer, device)
        self.gru_cell = GRUCell(input_dim, latent_dim)

    def __repr__(self):
        return "VanillaGRU"

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        return self.gru_cell(data, latent_state)


class DeltaTGRU(BaseRecurrentModel):
    """
        GRU by combining time gaps as input
    """

    def __init__(self, input_dim, latent_dim, eps_decay, decoder, timer, device):
        super(DeltaTGRU, self).__init__(input_dim, latent_dim, eps_decay, decoder, timer, device)
        # +1 dim for time gaps
        self.input_dim = input_dim + 1
        self.gru_cell = GRUCell(input_dim + 1, latent_dim).to(device)

    def __repr__(self):
        return "DeltaTGRU"

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        return self.gru_cell(torch.cat((data, dts.unsqueeze(-1)), dim=-1), latent_state)


class ExpDecayGRU(BaseRecurrentModel):
    """
        GRU with intermediate Exponential decay layer
    """

    def __init__(self, input_dim, latent_dim, eps_decay, decoder, timer, device):
        super(ExpDecayGRU, self).__init__(input_dim, latent_dim, eps_decay, decoder, timer, device)
        self.gru_cell = GRUCell(input_dim, latent_dim).to(device)
        self.decay_layer = nn.Linear(1, 1).to(device)

    def __repr__(self):
        return "ExpDecayGRU"

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        N = data.size(0)
        decay_coef = torch.exp(-torch.max(torch.zeros(N, 1, dtype=torch.float, device=self.device),
                                          self.decay_layer(dts.unsqueeze(-1))))
        assert decay_coef.size(0) == N
        assert decay_coef.size(1) == 1
        return self.gru_cell(data, decay_coef * latent_state)


class ODEGRU(BaseRecurrentModel):
    """
        GRU with intermediate ODE layer
    """

    def __init__(self, input_dim, latent_dim, eps_decay, decoder, diffeq_solver, timer, device):
        super(ODEGRU, self).__init__(input_dim, latent_dim, eps_decay, decoder, timer, device)
        self.diffeq_solver = diffeq_solver
        self.gru_cell = GRUCell(input_dim, latent_dim).to(device)

    def __repr__(self):
        return "ODEGRU"

    def encode_next_latent_state(self, data, latent_state, dts, odeint_rtol=None, odeint_atol=None, method=None):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        N = data.size(0)
        ts, inv_indices = torch.unique(dts, return_inverse=True)
        if ts[-1] == 0:
            return latent_state
        if ts[0] != 0:
            ts = torch.cat([torch.zeros(1, dtype=torch.float, device=self.device), ts])
            inv_indices += 1
        traj_latent_state = self.diffeq_solver(latent_state, ts, odeint_rtol, odeint_atol, method)
        selected_indices = tuple([torch.arange(N, dtype=torch.long, device=self.device), inv_indices])
        new_latent_state = traj_latent_state[selected_indices]  # [N, D_latent]
        assert new_latent_state.size(0) == N
        assert new_latent_state.size(1) == self.latent_dim
        return self.gru_cell(data, new_latent_state)


class BaseVAEModel(nn.Module):
    """
        Base VAE model as an abstract class
    """

    def __init__(self, input_dim, latent_dim, eps_decay, encoder_z0, decoder, timer, z0_prior, device):
        super(BaseVAEModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.encoder_z0 = encoder_z0
        self.decoder = decoder
        self.timer = timer
        self.z0_prior = z0_prior
        self.eps = 1.
        self.eps_decay = eps_decay
        self.i_step = 0
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "BaseVAEModel"

    def decay_eps(self):
        """
            Linear decay
        """
        if self.eps_decay > 0 and self.eps > 0:
            self.eps = max(0, 1. - self.eps_decay * self.i_step)
        self.i_step += 1

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        raise NotImplementedError("Abstract class cannot be used.")

    def encode_latent_traj(self, states, actions, time_steps, lengths, train=True):
        """
            Encode latent trajectories given states, actions and timesteps
            @:param states, shape [N, T, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T+1]
                    lengths, shape [N,]
            @:return hs, shape [N, T+1, D_latent]
        """
        N = states.size(0)
        if train:
            # encoding
            means_z0, logvars_z0 = self.encoder_z0(torch.cat((states, actions), dim=-1), time_steps, lengths)

            # reparam
            stds_z0 = torch.exp(0.5 * logvars_z0)
            eps = torch.randn_like(stds_z0)
            z0s = means_z0 + eps * stds_z0  # [N, D_latent]
        else:
            means_z0, stds_z0 = None, None
            z0s = self.sample_init_latent_states(num_trajs=N)

        # solve trajectory
        pred_next_states = []
        zs = [z0s]
        for i in range(time_steps.size(1) - 1):
            if i == 0 or (train and self.eps_decay == 0):
                data = torch.cat((states[:, i, :], actions[:, i, :]), dim=-1)  # [N, D_state+D_action]
            else:
                data = torch.cat((pred_next_states[-1], actions[:, i, :]), dim=-1)
                if train and self.eps > 0:  # scheduled sampling
                    heads = torch.rand(N) < self.eps  # [N,]
                    data[heads] = torch.cat((states[:, i, :], actions[:, i, :]), dim=-1)[heads]
            zs.append(self.encode_next_latent_state(data, zs[-1], time_steps[:, i + 1] - time_steps[:, i]))
            pred_next_states.append(self.decode_latent_traj(zs[-1]))
        zs = torch.stack(zs).permute(1, 0, 2)  # [T+1, N, D_latent]
        pred_next_states = torch.stack(pred_next_states).permute(1, 0, 2)  # [N, T, D_state]
        if train:
            self.decay_eps()

        assert zs.size(0) == N
        assert zs.size(1) == time_steps.size(1)
        assert zs.size(2) == self.latent_dim
        return zs, means_z0, stds_z0, pred_next_states

    def decode_latent_traj(self, zs):
        """
            Decode latent trajectories
            @:param zs, shape [N, T, D_latent]
            @:return shape [N, T, D_state]
        """
        return self.decoder(zs)

    def predict_next_states(self, states, actions, time_steps, lengths, train=True):
        """
            Predict next states given states, actions and timesteps
            @:param states, shape [N, T, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T+1]
                    lengths, shape [N,]
            @:return next_states, shape [N, T, D_state]
                     zs (current latent states), shape [N, T, D_latent]
                     mean_z,
                     std_z
        """
        # encoding and decoding
        zs, means_z0, stds_z0, next_states = self.encode_latent_traj(states, actions, time_steps, lengths,
                                                                     train=train)  # [N, T+1, D_latent]
        return next_states, zs[:, :-1, :], means_z0, stds_z0

    def compute_loss(self, states, actions, time_steps, lengths, dt_coef=.01, kl_coef=1., train=True):
        """
            Compute VAE's loss
            @:param states, shape [N, T+1, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T]
                    lengths, shape [N,]
                    dt_coef, the coefficient before dt loss term
                    kl_coef, the coefficient before kl term
            @:return total loss = mse + dt loss + kl_coef * kl_div
                     mse loss
        """
        # predict next states
        traj_cur_states, traj_next_states = states[:, :-1, :], states[:, 1:, :]
        traj_pred_next_states, traj_cur_latent_states, means_z0, stds_z0 = \
            self.predict_next_states(traj_cur_states, actions, time_steps, lengths, train=train)
        max_len = lengths.max()
        masks = torch.arange(max_len, device=self.device).expand(lengths.size(0), max_len) < lengths.unsqueeze(1)
        mse_loss = self.criterion(traj_next_states[masks], traj_pred_next_states[masks])

        # loss for time gap prediction
        if repr(self.timer) != 'Timer':
            dts = time_steps[:, 1:] - time_steps[:, :-1]
            dt_loss = self.timer.compute_loss(torch.cat((traj_cur_states, actions, traj_cur_latent_states), dim=-1),
                                              dts, masks)
        else:
            dt_loss = torch.tensor([0.], device=self.device)

        if train:
            # kl
            z0_dist = Normal(means_z0, stds_z0)
            kl_losses = kl_divergence(z0_dist, self.z0_prior)  # [N, D_latent]
            assert not torch.isnan(kl_losses).any()
            kl_loss = kl_losses.mean()
            return {'total': mse_loss + dt_coef * dt_loss + kl_coef * kl_loss, 'mse': mse_loss, 'dt': dt_loss,
                    'kl': kl_loss}
        else:
            return {'total': mse_loss + dt_coef * dt_loss, 'mse': mse_loss, 'dt': dt_loss}

    def sample_init_latent_states(self, num_trajs=0):
        shape = (self.latent_dim,) if num_trajs == 0 else (num_trajs, self.latent_dim)
        return self.z0_prior.sample(sample_shape=shape).squeeze(-1)


class VAEGRU(BaseVAEModel):
    """
        VAE with RNN encoder and RNN decoder
    """

    def __init__(self, input_dim, latent_dim, eps_decay, encoder_z0, decoder, timer, z0_prior, device):
        super(VAEGRU, self).__init__(input_dim, latent_dim, eps_decay, encoder_z0, decoder, timer, z0_prior, device)
        self.gru_cell = GRUCell(input_dim, latent_dim).to(device)

    def __repr__(self):
        return "VAEGRU"

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        return self.gru_cell(data, latent_state)


class LatentODE(BaseVAEModel):
    """
        Latent ODE
    """

    def __init__(self, input_dim, latent_dim, eps_decay, encoder_z0, decoder, diffeq_solver, timer, z0_prior, device):
        super(LatentODE, self).__init__(input_dim, latent_dim, eps_decay, encoder_z0, decoder, timer, z0_prior, device)
        self.diffeq_solver = diffeq_solver
        self.aug_layer = nn.Linear(input_dim + latent_dim, latent_dim).to(device)

    def __repr__(self):
        return "LatentODE"

    def encode_next_latent_state(self, data, latent_state, dts, odeint_rtol=None, odeint_atol=None, method=None):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        N = data.size(0)
        ts, inv_indices = torch.unique(dts, return_inverse=True)
        if ts[-1] == 0:
            return latent_state
        if ts[0] != 0:
            ts = torch.cat([torch.zeros(1, dtype=torch.float, device=self.device), ts])
            inv_indices += 1
        aug_latent_state = self.aug_layer(torch.cat((data, latent_state), dim=-1))
        traj_latent_state = self.diffeq_solver(aug_latent_state, ts, odeint_rtol, odeint_atol, method)
        selected_indices = tuple([torch.arange(N, dtype=torch.long, device=self.device), inv_indices])
        new_latent_state = traj_latent_state[selected_indices]  # [N, D_latent]
        assert new_latent_state.size(0) == N
        assert new_latent_state.size(1) == self.latent_dim
        return new_latent_state

    def rollout_timeline(self, data, latent_state, dts):
        aug_latent_state = self.aug_layer(torch.cat((data, latent_state), dim=-1))
        traj_latent_state = self.diffeq_solver(aug_latent_state, dts)
        return traj_latent_state  # [1, T, D]
