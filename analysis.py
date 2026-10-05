import numpy as np
import matplotlib.pyplot as plt
from environment import ForagingEnv
from agent_dynamics import DynamicAgent
from run_experiment import run, train_rl, mean_stay

if __name__ == "__main__":
    times = [2, 4, 6, 8, 10, 12]
    dyn, rl = [], []
    for T in times:
        d, r = [], []
        for seed in range(5):
            d.append(mean_stay(run(DynamicAgent(), ForagingEnv(travel_time=T, seed=seed), 1500)))
            ag = train_rl(ForagingEnv(travel_time=T, seed=100 + seed))
            r.append(mean_stay(run(ag, ForagingEnv(travel_time=T, seed=200 + seed), 1500, learn=False)))
        dyn.append((np.mean(d), np.std(d))); rl.append((np.mean(r), np.std(r)))
        print(T, dyn[-1], rl[-1])
    for name, v in [("dynamical", dyn), ("Q-learning", rl)]:
        m, s = np.array(v).T
        plt.errorbar(times, m, s, marker="o", capsize=3, label=name)
    plt.xlabel("travel time (ticks)"); plt.ylabel("mean stay per patch (ticks)")
    plt.legend(); plt.title("Longer travel -> longer stay (5 seeds)")
    plt.savefig("travel_sweep.png", dpi=150); plt.show()