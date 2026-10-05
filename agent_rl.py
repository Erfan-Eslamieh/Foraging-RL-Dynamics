import numpy as np

ACTIONS = ["stay", "leave"]


class QLearningAgent:
    """Tabular Q over (state, action). State = ticks in patch, or travel countdown."""

    def __init__(self, n_states, lr=0.1, gamma=0.95, epsilon=0.2, seed=0):
        self.q = np.zeros((n_states, 2))
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.n_patch_states = None

    def decide(self, state):
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(2))
        else:
            a = int(np.argmax(self.q[state]))
        return ACTIONS[a]

    def update(self, r, mode, state=None, action=None, next_state=None, **kw):
        a = ACTIONS.index(action)
        target = r + self.gamma * self.q[next_state].max()
        self.q[state, a] += self.lr * (target - self.q[state, a])