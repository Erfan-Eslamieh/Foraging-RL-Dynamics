import numpy as np
from environment import ForagingEnv
from agent_dynamics import DynamicAgent
from agent_rl import QLearningAgent

N_MAX = 30


def run(agent, env, steps=300, learn=True):
    s = env.reset(); hist = []
    for t in range(steps):
        s = env.state(N_MAX)
        a = agent.decide(s)
        r, mode = env.step(a)
        s2 = env.state(N_MAX)
        if learn:
            agent.update(r, mode, state=s, action=a, next_state=s2)
        hist.append(dict(r=r, mode=mode, x=getattr(agent, "x", np.nan),
                         rbar=getattr(agent, "rbar", np.nan),
                         pos=env.cur, prev=env.prev, target=getattr(env, "target", env.cur), left=env.travel_left,
                         tot=env.travel_time, food=env.reward))
    return hist


def train_rl(env, steps=60000, seed=0):
    ag = QLearningAgent(N_MAX + env.travel_time + 1, seed=seed)
    run(ag, env, steps)
    ag.epsilon = 0.0
    return ag


def mean_stay(hist):
    """Average ticks spent per patch visit (mean length of exploit runs)."""
    runs, c = [], 0
    for h in hist:
        if h["mode"] == "exploit":
            c += 1
        elif c:
            runs.append(c); c = 0
    return np.mean(runs) if runs else np.nan


def total_reward(hist):
    return sum(h["r"] for h in hist)


if __name__ == "__main__":
    env = ForagingEnv()
    h = run(DynamicAgent(), env, 300)
    print("dynamic: stay", mean_stay(h), "reward", total_reward(h))
    rl = train_rl(ForagingEnv(seed=1))
    h = run(rl, ForagingEnv(seed=2), 300, learn=False)
    print("RL: stay", mean_stay(h), "reward", total_reward(h))