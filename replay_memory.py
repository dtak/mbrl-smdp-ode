import math
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'latent_state', 'next_latent_state', 'dt'))

Trajectory = namedtuple('Trajectory', ('states', 'actions', 'time_steps', 'length'))


class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
        self.tree_size = 2 ** self.tree_level - 1
        self.tree = [0 for _ in range(self.tree_size)]
        self.data = [None for _ in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2 ** (self.tree_level - 1) - 1 <= index:
            return self.data[index - (2 ** (self.tree_level - 1) - 1)], self.tree[index], index - (
                        2 ** (self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]

        if value <= left:
            return self._find(value, 2 * index + 1)
        else:
            return self._find(value - left, 2 * (index + 1))

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size


class ReplayMemory(object):
    """
        Replay buffer
    """

    def __init__(self, capacity, tuple):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.tuple = tuple

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.tuple(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    """ The class represents prioritized experience replay buffer.
    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.
    see https://arxiv.org/pdf/1511.05952.pdf .
    """

    def __init__(self, capacity, tuple, alpha=0.6, beta=0.4):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        beta : float
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.tuple = tuple

    def push(self, *args, priority):
        """ Add new sample.

        Parameters
        ----------
        args : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(self.tuple(*args), priority ** self.alpha)

    def sample(self, batch_size):
        """ The method return samples randomly.

        Parameters
        ----------
        batch_size : int
            sample size to be extracted

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        i = 0
        while i < batch_size:
            r = random.random()
            data, priority, index = self.tree.find(r)
            if not data:
                continue
            priorities.append(priority)
            weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0])  # To avoid duplicating
            i += 1

        self.priority_update(indices, priorities)  # Revert priorities

        weights = [w / max(weights) for w in weights]  # Normalize for stability

        return out, weights, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

    def __len__(self):
        return self.tree.filled_size()
