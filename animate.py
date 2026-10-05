import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.widgets import Slider, Button
from environment import ForagingEnv
from agent_dynamics import DynamicAgent
from run_experiment import run

STEPS, WIN = 200, 60
XS = [1.0, 3.0, 5.0]                 # patch x positions
SAVE_GIF = False                     # True: also save foraging.gif (slow)
hist = run(DynamicAgent(), ForagingEnv(seed=3), STEPS)

# ---------- figure ----------
fig = plt.figure(figsize=(10, 8.5))
ax = fig.add_axes([0.05, 0.50, 0.90, 0.46])
ay = fig.add_axes([0.09, 0.22, 0.86, 0.24])

# ---------- scene ----------
ax.set_xlim(0, 6); ax.set_ylim(0, 3); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 6, 3, color="#dff0f7"))
ax.add_patch(Rectangle((0, 0), 6, 1.05, color="#bfdca0"))
for x in XS:
    ax.add_patch(Circle((x, 1.55), 0.75, color="#3f8f4a", zorder=2))
    ax.add_patch(Rectangle((x - 0.06, 0.85), 0.12, 0.35, color="#6b4a2b", zorder=1))
berries = ax.scatter([], [], s=90, color="#d2342b", edgecolors="#7a1511", zorder=3)
labels = [ax.text(x, 2.45, "", ha="center", fontsize=11, weight="bold") for x in XS]
title = ax.text(0.1, 2.75, "", fontsize=17, weight="bold")
ax.text(5.9, 2.75, "Exploit = stay and eat\nExplore = walk to a new patch",
        ha="right", va="top", fontsize=10, color="#444")

# squirrel: parts defined relative to its position (dx, dy, kind)
SC = 1.5
parts = {
    "tail": Ellipse((0, 0), 0.26 * SC, 0.62 * SC, color="#c97a2b", zorder=5),
    "body": Ellipse((0, 0), 0.46 * SC, 0.32 * SC, color="#a55a1a", zorder=6),
    "belly": Ellipse((0, 0), 0.22 * SC, 0.17 * SC, color="#f0d9b5", zorder=7),
    "head": Circle((0, 0), 0.12 * SC, color="#a55a1a", zorder=8),
    "ear": Circle((0, 0), 0.045 * SC, color="#7a3f10", zorder=7),
    "eye": Circle((0, 0), 0.025 * SC, color="black", zorder=9),
    "nose": Circle((0, 0), 0.02 * SC, color="#222", zorder=9),
}
for p in parts.values():
    ax.add_patch(p)


def place_squirrel(x, y, face, wobble):
    s = face
    off = {"tail": (-s * 0.30, 0.20), "body": (0, 0), "belly": (s * 0.05, -0.03),
           "head": (s * 0.25, 0.15), "ear": (s * 0.22, 0.29), "eye": (s * 0.30, 0.19),
           "nose": (s * 0.37, 0.15)}
    for k, (dx, dy) in off.items():
        parts[k].center = (x + dx * SC, y + dy * SC)
    parts["tail"].angle = s * (22 + wobble)


# ---------- plot (drawn once, window slides) ----------
idx = np.arange(STEPS)
ay.set_ylim(-3, 17); ay.set_ylabel("value", fontsize=11)
ay.set_xlabel("time (ticks)", fontsize=11)
for j, h in enumerate(hist):
    ay.axvspan(j, j + 1, ymin=0, ymax=0.06,
               color="tab:green" if h["mode"] == "exploit" else "tab:orange")
ay.axhline(0, color="k", alpha=.2)
lr, = ay.plot([], [], color="gray", lw=1.2, label="reward")
lx, = ay.plot([], [], color="tab:blue", lw=2.8, label="x (fast state)")
lb, = ay.plot([], [], color="tab:orange", lw=2.2, ls="--", label="rbar (slow avg rate)")
ay.legend(loc="upper right", ncol=3, fontsize=9)
ay.text(0.01, 0.08, "green = exploitation   orange = exploration", transform=ay.transAxes,
        fontsize=9, va="bottom", color="#333")

# ---------- precompute berries shown at each tick ----------
def food_at(i):
    h = hist[i]
    if h["mode"] == "exploit":
        return {h["pos"]: h["food"]}
    left = next((hist[j]["food"] for j in range(i - 1, -1, -1) if hist[j]["mode"] == "exploit"), 0)
    fresh = next((hist[j]["r"] for j in range(i + 1, STEPS) if hist[j]["mode"] == "exploit"), None)
    d = {h["prev"]: left}
    if fresh is not None:
        d[h["target"]] = fresh
    return d


FOOD = [food_at(i) for i in range(STEPS)]
state = {"face": 1}


def draw(t):
    i = min(int(t), STEPS - 1); f = t - int(t); h = hist[i]
    if h["mode"] == "exploit":
        x, moving = XS[h["pos"]], False
    else:
        k = (h["tot"] - h["left"] + f) / (h["tot"] + 1)
        x0, x1 = XS[h["prev"]], XS[h["target"]]
        x, moving = x0 + (x1 - x0) * k, True
        state["face"] = 1 if x1 > x0 else -1
    bob = 0.07 * abs(np.sin(t * 6)) if moving else 0.0
    place_squirrel(x, 0.62 + bob, state["face"], 8 * np.sin(t * 5) if moving else 0)
    pts = []
    for lab, px_ in zip(labels, range(3)):
        n = FOOD[i].get(px_)
        lab.set_text("" if n is None else f"food {n:.1f}")
        if n is not None:
            m = max(0, min(int(round(n)), 18))
            for j in range(m):
                r = 0.55 * np.sqrt((j + .5) / 18); a = j * 2.4
                pts.append((XS[px_] + r * np.cos(a), 1.55 + r * np.sin(a)))
    berries.set_offsets(np.array(pts) if pts else np.empty((0, 2)))
    title.set_text("EXPLOITING (eating)" if h["mode"] == "exploit" else "EXPLORING (walking)")
    title.set_color("tab:green" if h["mode"] == "exploit" else "tab:orange")
    sl = slice(0, i + 1)
    for line, key in [(lr, "r"), (lx, "x"), (lb, "rbar")]:
        line.set_data(idx[sl], [q[key] for q in hist[sl]])
    ay.set_xlim(max(0, i - WIN + 1), max(WIN, i + 1))


# ---------- manual controls ----------
s_exploit = Slider(fig.add_axes([0.20, 0.12, 0.28, 0.03]), "exploit speed", 0.3, 10, valinit=2.0)
s_explore = Slider(fig.add_axes([0.68, 0.12, 0.28, 0.03]), "explore speed", 0.3, 10, valinit=2.0)
btn = Button(fig.add_axes([0.40, 0.04, 0.09, 0.05]), "Pause")
btn2 = Button(fig.add_axes([0.51, 0.04, 0.09, 0.05]), "Restart")
ctl = {"t": 0.0, "pause": False, "last": time.perf_counter()}
def toggle(e):
    ctl["pause"] = not ctl["pause"]
    btn.label.set_text("Play" if ctl["pause"] else "Pause")


btn.on_clicked(toggle)
btn2.on_clicked(lambda e: ctl.update(t=0.0))


def update(_):
    now = time.perf_counter(); dt = min(now - ctl["last"], 0.1); ctl["last"] = now
    if not ctl["pause"]:
        mode = hist[min(int(ctl["t"]), STEPS - 1)]["mode"]
        ctl["t"] += (s_exploit.val if mode == "exploit" else s_explore.val) * dt   # ticks per second
        if ctl["t"] >= STEPS - 1:
            ctl["t"] = 0.0
    draw(ctl["t"])


anim = FuncAnimation(fig, update, frames=itertools.count(), interval=30,
                     blit=False, cache_frame_data=False)

if __name__ == "__main__":
    if SAVE_GIF:
        ctl["pause"] = True
        gif = FuncAnimation(fig, lambda t: draw(t), frames=np.arange(0, STEPS - 1, 0.12))
        gif.save("foraging.gif", writer="pillow", fps=15)
        ctl["pause"] = False
    plt.show()