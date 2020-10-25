"""HIV Treatment domain based on https://bitbucket.org/rlpy/rlpy/src/master/rlpy/Domains/HIVTreatment.py
Deborah Hanus of Harvard DTAK contributed to the implementation.
"""
import numpy as np
from scipy.integrate import ode
from collections.abc import Iterable

# Original attribution information:
__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class HIVSimulator(object):
    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """
    state_names = ("T1", "T2", "T1*", "T2*", "V", "E")
    eps_values_for_actions = np.array([[0., 0.], [.7, 0.], [0., .3], [.7, .3]])
    binary_action_codes = [(0, 0), (1, 0), (0, 1), (1, 1)]
    baseline_state = np.array([163573., 5., 11945., 46., 63919., 24.])
    empirical_mean = np.array([5.41947795, 1.59751311, 3.03701108, 1.45943164, 3.7256933, 1.73064582])
    empirical_std = np.array([0.1913442, 0.86731865, 1.19517352, 0.48159324, 1.16411527, 0.20234819])

    def __init__(self, logspace=True, model_derivatives=None, perturb_params=True, podmp=False, p_init=np.zeros(6),
                 p_T1=0, p_T2=0, p_T1s=0, p_T2s=0, p_V=0, p_E=0, **kw):
        """
        Initialize the environment.

        Keyword arguments:
        logspace --  return the state as log(state)
        dt -- change in time for each action (in days)
        model_derivatives -- option to pass specific model derivatives
        perturb_params -- boolean indicating whether to perturb the initial state
        p_T1 -- initial perturbation factor for specific state dimension
        p_T2 -- initial perturbation factor for specific state dimension
        p_T1s -- initial perturbation factor for specific state dimension
        p_T2s -- initial perturbation factor for specific state dimension
        p_V -- initial perturbation factor for specific state dimension
        p_E -- initial perturbation factor for specific state dimension
        """
        self.logspace = logspace
        if logspace:
            self.statespace_limits = np.array([[-5, 8]] * 6)
        else:
            self.statespace_limits = np.array([[0., 1e8]] * 6)
        if model_derivatives is None:
            model_derivatives = dsdt
        self.model_derivatives = model_derivatives
        self.pomdp = podmp
        self.reward_bound = 1e300
        self.num_actions = 4
        self.num_states = 4 if podmp else 6
        self.mask = [True, True, False, False, True, True] if podmp else [True for _ in range(6)]
        self.perturb_params = ('p_lambda1', 'p_lambda2', 'p_k1', 'p_k2', 'p_f', \
                               'p_m1', 'p_m2', 'p_lambdaE', 'p_bE', 'p_Kb', 'p_d_E', 'p_Kd')
        self.p_init = p_init
        self.reset(perturb_params)

    def __repr__(self):
        return "HIV_Simulator"

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, perturb_params=True):
        """Reset the environment."""
        self.t = 0
        # non-healthy stable state of the system
        if not perturb_params:
            self.state = self.baseline_state
        else:
            self.state = self.baseline_state + (self.baseline_state * self.p_init)
        return (self._get_ob() - self.empirical_mean[self.mask]) / self.empirical_std[self.mask]

    def _get_ob(self):
        """Return current state."""
        if self.logspace:
            return np.log10(self.state)[self.mask]
        else:
            return self.state[self.mask]

    def calc_reward(self, action=0, state=None, dt=5):
        """Calculate the reward for the specified transition."""
        try:
            if isinstance(action, float):
                action = int(action)
            elif isinstance(action, Iterable):
                action = self.binary_action_codes.index(tuple(action))
            assert (action in [0, 1, 2, 3])
        except:
            raise ValueError("Must pass an action in [0,1,2,3] or [[0,0],[0,1],[1,0],[1,1]]")

        eps1, eps2 = self.eps_values_for_actions[action]
        if state is None:
            state = self._get_ob()
        else:  # recover from the standardlized state
            state = state * self.empirical_std[self.mask] + self.empirical_mean[self.mask]
        if self.pomdp:
            if self.logspace:
                _, _, V, E = 10 ** state
            else:
                _, _, V, E = state
        else:
            if self.logspace:
                _, _, _, _, V, E = 10 ** state
            else:
                _, _, _, _, V, E = state
        # the reward function penalizes treatment because of side-effects
        reward = -0.1 * V - 2e4 * eps1 ** 2 - 2e3 * eps2 ** 2 + 1e3 * E
        # Constrain reward to be within specified range
        if np.isnan(reward) or np.isinf(reward):
            reward = -self.reward_bound
        elif reward > self.reward_bound:
            reward = self.reward_bound
        elif reward < -self.reward_bound:
            reward = -self.reward_bound
        return reward

    def encode_action(self, action):
        try:
            if isinstance(action, float):
                action = int(action)
            elif isinstance(action, Iterable):
                action = self.binary_action_codes.index(tuple(action))
            assert (action in [0, 1, 2, 3])
        except:
            raise ValueError("Must pass an action in [0,1,2,3] or [[0,0],[0,1],[1,0],[1,1]]")
        return self.binary_action_codes[action]

    def step(self, action, dt=5, perturb_params=False, p_lambda1=0, p_lambda2=0, p_k1=0,
             p_k2=0, p_f=0, p_m1=0, p_m2=0, p_lambdaE=0, p_bE=0, p_Kb=0, p_d_E=0, p_Kd=0, **kw):
        """Perform the specified action and update the environment.

        Arguments:
        action -- action to be taken. Can be an index from 0-3 or an element of `binary_action_codes`

        Keyword Arguments:
        perturb_params -- boolean indicating whether to perturb dynamics (default: False)
        p_lambda1 -- hidden parameter (default: 0)
        p_lambda2 -- hidden parameter (default: 0)
        p_k1 -- hidden parameter (default: 0)
        p_k2 -- hidden parameter (default: 0)
        p_f -- hidden parameter (default: 0)
        p_m1 -- hidden parameter (default: 0)
        p_m2 -- hidden parameter (default: 0)
        p_lambdaE -- hidden parameter (default: 0)
        p_bE -- hidden parameter (default: 0)
        p_Kb -- hidden parameter (default: 0)
        p_d_E -- hidden parameter (default: 0)
        p_Kd -- hidden parameter (default: 0)
        """

        try:
            if isinstance(action, float):
                action = int(action)
            elif isinstance(action, Iterable):
                action = self.binary_action_codes.index(tuple(action))
            assert (action in [0, 1, 2, 3])
        except:
            raise ValueError("Must pass an action in [0,1,2,3] or [[0,0],[0,1],[1,0],[1,1]]")

        self.t += dt
        self.action = action
        eps1, eps2 = self.eps_values_for_actions[action]
        r = ode(self.model_derivatives).set_integrator('vode', nsteps=int(2000 * dt), method='bdf')
        t0 = 0
        deriv_args = (
        eps1, eps2, perturb_params, p_lambda1, p_lambda2, p_k1, p_k2, p_f, p_m1, p_m2, p_lambdaE, p_bE, p_Kb, p_d_E,
        p_Kd)
        r.set_initial_value(self.state, t0).set_f_params(deriv_args)
        self.state = r.integrate(dt)
        reward = self.calc_reward(action=action)
        return (self._get_ob() - self.empirical_mean[self.mask]) / self.empirical_std[self.mask], reward, \
               self.is_terminal(), {}

    def is_terminal(self, state=None):
        return self.t >= 1000

    def get_time_gap(self, action=0, state=None):
        if state is None:
            state = self._get_ob()
        else:  # recover from the standardlized state
            state = state * self.empirical_std[self.mask] + self.empirical_mean[self.mask]
        if self.pomdp:
            if self.logspace:
                _, _, V, _ = 10 ** state
            else:
                _, _, V, _ = state
        else:
            if self.logspace:
                _, _, _, _, V, _ = 10 ** state
            else:
                _, _, _, _, V, _ = state
        if V <= 1e4:  # healthy
            if action == 0:
                t = np.random.randint(7, 15)
            else:
                t = np.random.randint(3, 8)
        elif 1e4 < V <= 1e5:  # unhealthy, need notice
            if action == 0:
                t = np.random.randint(3, 8)
            elif action == 1 or action == 2:
                t = np.random.randint(3, 6)
            else:
                t = 3
        else:
            if action == 0:  # unhealthy, need clinical observation
                t = 3
            else:
                t = np.random.randint(1, 3)
        return t

    def get_time_info(self):
        return 1, 14, 1000, False  # min_t, max_t, max time length, is continuous


def dsdt(t, s, params):
    """Wrapper for system derivative with respect to time"""
    derivs = np.empty_like(s)
    eps1, eps2, perturb_params, p_lambda1, p_lambda2, p_k1, p_k2, p_f, p_m1, p_m2, p_lambdaE, p_bE, p_Kb, p_d_E, p_Kd = params
    dsdt_(derivs, s, t, eps1, eps2, perturb_params, p_lambda1, p_lambda2, p_k1, \
          p_k2, p_f, p_m1, p_m2, p_lambdaE, p_bE, p_Kb, p_d_E, p_Kd)
    return derivs


def dsdt_(out, s, t, eps1, eps2, perturb_params=False, p_lambda1=0, p_lambda2=0, p_k1=0, \
          p_k2=0, p_f=0, p_m1=0, p_m2=0, p_lambdaE=0, p_bE=0, p_Kb=0, p_d_E=0, p_Kd=0):
    """System derivate with respect to time (days).

    Arguments:
    out -- output
    s -- state
    t -- time
    eps1 -- action effect
    eps2 -- action effect
    """
    # baseline model parameter constants
    lambda1 = 1e4  # Target cell, type 1, production rate *CAN BE VARIED*
    lambda2 = 31.98  # Target cell, type 2, production rate *CAN BE VARIED*
    d1 = 0.01  # Target cell, type 1, death rate
    d2 = 0.01  # Target cell, type 2, death rate
    f = .34  # Treatment efficacy, reduction in population 2 \in[0,1] *CAN BE VARIED*
    k1 = 8e-7  # Population 1, infection rate, *SENSITIVE TO REDUCTION, CAN BE VARIED*
    k2 = 1e-4  # Population 2, infection rate, *SENSITIVE TO REDUCTION, CAN BE VARIED*
    delta = .7  # Infected cell death rate
    m1 = 1e-5  # Immune-induced clearance rate, population 1 *CAN BE VARIED*
    m2 = 1e-5  # Immune-induced clearance rate, population 2 *CAN BE VARIED*
    NT = 100.  # Virions produced per infected cell
    c = 13.  # Virius natural death rate
    rho1 = 1.  # Average number of virions infecting type 1 cell
    rho2 = 1.  # Average number of virions infecting type 2 cell
    lambdaE = 1.  # Immune effector production rate *CAN BE VARIED*
    bE = 0.3  # Maximum birth rate for immune effectors *SENSITVE TO GROWTH, CAN BE VARIED*
    Kb = 100.  # Saturation constant for immune effector birth *CAN BE VARIED*
    d_E = 0.25  # Maximum death rate for immune effectors *CAN BE VARIED*
    Kd = 500.  # Saturation constant for immune effectors death *CAN BE VARIED*
    deltaE = 0.1  # Natural death rate for immune effectors

    if perturb_params:
        # Perturb empirically varied parameters...
        d = 1  # Scaling factor
        lambda1 += lambda1 * (p_lambda1 * d)
        lambda2 += lambda2 * (p_lambda2 * d)
        k1 += k1 * (p_k1 * d)
        k2 += k2 * (p_k2 * d)
        f += f * (p_f * d)
        m1 += m1 * (p_m1 * d)
        m2 += m2 * (p_m2 * d)
        lambdaE += lambdaE * (p_lambdaE * d)
        bE += bE * (p_bE * d)
        Kb += Kb * (p_Kb * d)
        d_E += d_E * (p_d_E * d)
        Kd += Kd * (p_Kd * d)

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1. - eps1) * k1 * V * T1
    tmp2 = (1. - f * eps1) * k2 * V * T2
    out[0] = lambda1 - d1 * T1 - tmp1
    out[1] = lambda2 - d2 * T2 - tmp2
    out[2] = tmp1 - delta * T1s - m1 * E * T1s
    out[3] = tmp2 - delta * T2s - m2 * E * T2s
    out[4] = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
             - ((1. - eps1) * rho1 * k1 * T1 +
                (1. - f * eps1) * rho2 * k2 * T2) * V
    out[5] = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
             - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E


try:
    import numba
except ImportError as e:
    print("Numba acceleration unavailable, expect slow runtime.")
else:
    dsdt_ = numba.jit(
        numba.void(numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.float64, numba.bool_, \
                   numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, \
                   numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float64),
        nopython=True, nogil=True)(dsdt_)
