import numpy as np


class ForagingEnv:
    """Three patches in a row. Staying gives a decaying reward.
    Leaving costs energy and takes travel_time ticks to reach a fresh patch."""

    def __init__(self, decay=0.9, travel_cost=1.5, travel_time=5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.decay = decay
        self.travel_cost = travel_cost
        self.travel_time = travel_time
        self.reset()

    def _new_patch(self):
        self.reward = self.rng.uniform(6, 15)   # patches differ in richness
        self.n = 0                              # ticks spent in this patch

    def reset(self):
        self.cur, self.prev = 0, 0
        self.travel_left = 0
        self._new_patch()
        return self.state()

    def state(self, n_max=30):
        # patch states 0..n_max-1, travel states n_max..n_max+travel_time
        if self.travel_left > 0:
            return n_max + self.travel_left
        return min(self.n, n_max - 1)

    def step(self, action):
        if self.travel_left > 0:                 # action is ignored while walking
            self.travel_left -= 1
            if self.travel_left == 0:
                self.cur = self.target
                self._new_patch()
            return 0.0, "explore"
        if action == "leave":
            self.prev = self.cur
            self.target = int(self.rng.choice([i for i in range(3) if i != self.cur]))
            self.travel_left = self.travel_time
            return -self.travel_cost, "explore"
        r = self.reward
        self.reward *= self.decay
        self.n += 1
        return r, "exploit"