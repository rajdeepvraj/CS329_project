# app.py — Flood Escape RL interactive (full-featured)
# Save as app.py and run: streamlit run app.py

import streamlit as st
st.set_page_config(layout="wide", page_title="Flood Escape RL — Full Demo")

# === Installs/imports (assume packages installed) ===
import numpy as np
import random
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from tqdm import trange
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# === Utility plotting defaults ===
# Be robust about style: try seaborn, fallback to matplotlib builtin
try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except Exception:
    try:
        plt.style.use("seaborn-darkgrid")
    except Exception:
        plt.style.use("ggplot")

# === Environment (supports all flood modes) ===
class FloodGridWorld:
    """
    Grid world supporting multiple flood behaviors:
      modes:
        'static'       : no expansion, only initial flooded cells (for testing)
        'global_rise'  : global water level rises every step
        'bfs_spread'   : BFS wavefront from source(s)
        'terrain_bfs'  : BFS with terrain-influenced delays
        'stochastic'   : terrain_bfs + random delays + rainfall bursts
        'multi_bfs'    : multi-source BFS (fast multi-breach)
    Actions: up=0,right=1,down=2,left=3
    """
    def __init__(self, N=8, start=None, goal=None, flood_mode="terrain_bfs", seed=None,
                 water_rise_per_step=0.02, bfs_base_delay=1, terrain_friction=0.5,
                 random_delay_prob=0.15, rainfall_burst_prob=0.02, flood_sources=None,
                 max_steps=200, reward_goal=100.0, reward_drown=-50.0, reward_step=-1.0):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.H = self.W = int(N)
        self.start = start if start is not None else (self.H-1, 0)
        self.goal = goal if goal is not None else (0, self.W-1)
        self.flood_mode = flood_mode
        self.max_steps = int(max_steps)
        self.water_rise_per_step = float(water_rise_per_step)
        self.bfs_base_delay = int(bfs_base_delay)
        self.terrain_friction = float(terrain_friction)
        self.random_delay_prob = float(random_delay_prob)
        self.rainfall_burst_prob = float(rainfall_burst_prob)
        self.flood_sources = flood_sources if flood_sources is not None else [(self.H-1, 0)]
        # elevation: gradient + noise
        base = np.linspace(0.2, 1.0, self.H)[:, None] + np.linspace(0, 0.4, self.W)[None, :]
        self.elev = np.clip(base + 0.05*np.random.randn(self.H, self.W), 0.0, 2.0)
        self.reward_goal = float(reward_goal)
        self.reward_drown = float(reward_drown)
        self.reward_step = float(reward_step)
        # episode-state
        self.reset(initialize=True)

    def reset(self, initialize=False, random_start=False):
        # initialize flood structures when requested
        if initialize:
            # keep elevation same; flood sources default if not set
            if len(self.flood_sources) == 0:
                self.flood_sources = [(self.H-1, 0)]
        self.t = 0
        self.water = 0.0
        self.pos = tuple(self.start) if not random_start else (random.randrange(self.H), random.randrange(self.W))
        self.done = False
        # BFS flood arrival times (in steps)
        self.flood_time = np.full((self.H, self.W), np.inf)
        self._front = []  # frontier of (r,c,arrival_time)
        if self.flood_mode in ("bfs_spread", "terrain_bfs", "stochastic", "multi_bfs"):
            for (r,c) in self.flood_sources:
                if 0 <= r < self.H and 0 <= c < self.W:
                    self.flood_time[r,c] = 0
                    self._front.append((r,c,0))
        return self._state()

    def _state(self):
        if self.flood_mode == "global_rise":
            return (self.pos[0], self.pos[1], round(self.water,4))
        else:
            return (self.pos[0], self.pos[1], int(self.t))

    def _in_bounds(self, r,c):
        return 0 <= r < self.H and 0 <= c < self.W

    def _propagate_bfs_step(self):
        # expand frontier entries whose assigned time == current time
        new_nodes = []
        for (r,c,ti) in list(self._front):
            if ti != self.t:
                continue
            for (nr,nc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if not self._in_bounds(nr,nc): continue
                if self.flood_time[nr,nc] <= self.t: continue
                delay = self.bfs_base_delay
                if self.flood_mode in ("terrain_bfs","stochastic"):
                    elev_ratio = (self.elev[nr,nc] / (np.max(self.elev)+1e-9))
                    delay = max(1, int(round(self.bfs_base_delay * (1 + self.terrain_friction * elev_ratio))))
                if self.flood_mode == "stochastic" and random.random() < self.random_delay_prob:
                    delay += 1
                arrival_time = self.t + delay
                if arrival_time < self.flood_time[nr,nc]:
                    self.flood_time[nr,nc] = arrival_time
                    new_nodes.append((nr,nc,arrival_time))
        self._front.extend(new_nodes)

    def _maybe_rain(self, bias_to_low=True):
        # placeholder for future rainfall behaviour
        if self.flood_mode != "stochastic":
            return
        # pick low elevation cell with bias
        flat = [(i,j) for i in range(self.H) for j in range(self.W)]
        weights = (np.max(self.elev) - self.elev).flatten() + 1e-6
        idx = random.choices(flat, weights=weights, k=1)[0]
        rr,cc = idx
        self.flood_time[rr,cc] = min(self.flood_time[rr,cc], self.t)
        self._front.append((rr,cc,self.t))

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        r,c = self.pos
        if action == 0: nr,nc = r-1,c
        elif action == 1: nr,nc = r,c+1
        elif action == 2: nr,nc = r+1,c
        elif action == 3: nr,nc = r,c-1
        else: nr,nc = r,c
        if not self._in_bounds(nr,nc):
            nr,nc = r,c
        self.pos = (nr,nc)
        self.t += 1
        # advance flood AFTER movement (agent experiences new flood immediately)
        if self.flood_mode == "global_rise":
            self.water += self.water_rise_per_step
        else:
            # stochastic rainfall: randomly flood a cell at current time occasionally
            if self.flood_mode == "stochastic" and random.random() < self.rainfall_burst_prob:
                # pick low elevation cell (higher chance)
                flat = [(i,j) for i in range(self.H) for j in range(self.W)]
                weights = (np.max(self.elev) - self.elev).flatten() + 1e-6
                idx = random.choices(flat, weights=weights, k=1)[0]
                rr,cc = idx
                self.flood_time[rr,cc] = min(self.flood_time[rr,cc], self.t)
                self._front.append((rr,cc,self.t))
            # also call maybe_rain to keep consistent behavior
            if self.flood_mode == "stochastic":
                self._maybe_rain()
            self._propagate_bfs_step()
        # event checks
        elev_here = float(self.elev[nr,nc])
        if (nr,nc) == self.goal:
            self.done = True
            return self._state(), float(self.reward_goal), True, {"event":"goal"}
        drowned = False
        if self.flood_mode == "global_rise":
            if self.water > elev_here:
                drowned = True
        else:
            if self.flood_time[nr,nc] <= self.t:
                drowned = True
        if drowned:
            self.done = True
            return self._state(), float(self.reward_drown), True, {"event":"drowned"}
        if self.t >= self.max_steps:
            self.done = True
            return self._state(), float(-1.0), True, {"event":"timeout"}
        return self._state(), float(self.reward_step), False, {"event":"continue"}

    def sample_action(self):
        return random.randrange(4)

    def state_to_index(self, state):
        return int(state[0])*self.W + int(state[1])

    def all_states(self):
        return [(r,c) for r in range(self.H) for c in range(self.W)]

# === RL helpers and trainers ===
def ensure_Q_entry(Q, s_idx, n_actions):
    if s_idx not in Q:
        Q[s_idx] = [0.0]*n_actions

def epsilon_greedy(Q, s_idx, n_actions, eps):
    if random.random() < eps:
        return random.randrange(n_actions)
    if s_idx not in Q:
        return random.randrange(n_actions)
    arr = Q[s_idx]
    maxv = max(arr)
    choices = [i for i,v in enumerate(arr) if v==maxv]
    return random.choice(choices)

def train_q_learning(env, episodes=1500, alpha=0.3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    nA = 4
    Q = {}
    rewards = []
    eps = eps_start
    for ep in trange(episodes, desc="Q-Learning"):
        state = env.reset()
        s_idx = env.state_to_index(state)
        ensure_Q_entry(Q, s_idx, nA)
        total_r = 0.0
        done = False
        while not done:
            a = epsilon_greedy(Q, s_idx, nA, eps)
            next_state, r, done, _ = env.step(a)
            ns_idx = env.state_to_index(next_state)
            ensure_Q_entry(Q, ns_idx, nA)
            Q[s_idx][a] += alpha * (r + gamma * max(Q[ns_idx]) - Q[s_idx][a])
            total_r += r
            s_idx = ns_idx
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    return Q, rewards

def train_sarsa(env, episodes=1500, alpha=0.3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    nA = 4
    Q = {}
    rewards = []
    eps = eps_start
    for ep in trange(episodes, desc="SARSA"):
        state = env.reset()
        s_idx = env.state_to_index(state)
        ensure_Q_entry(Q, s_idx, nA)
        a = epsilon_greedy(Q, s_idx, nA, eps)
        total_r = 0.0
        done = False
        while not done:
            next_state, r, done, _ = env.step(a)
            ns_idx = env.state_to_index(next_state)
            ensure_Q_entry(Q, ns_idx, nA)
            if done:
                Q[s_idx][a] += alpha * (r - Q[s_idx][a])
                total_r += r
                break
            a2 = epsilon_greedy(Q, ns_idx, nA, eps)
            Q[s_idx][a] += alpha * (r + gamma * Q[ns_idx][a2] - Q[s_idx][a])
            total_r += r
            s_idx = ns_idx
            a = a2
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    return Q, rewards

def train_double_q(env, episodes=1500, alpha=0.3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    nA = 4
    Q1 = {}
    Q2 = {}
    rewards = []
    eps = eps_start
    for ep in trange(episodes, desc="Double-Q"):
        state = env.reset()
        s_idx = env.state_to_index(state)
        ensure_Q_entry(Q1, s_idx, nA); ensure_Q_entry(Q2, s_idx, nA)
        total_r = 0.0
        done = False
        while not done:
            ensure_Q_entry(Q1, s_idx, nA); ensure_Q_entry(Q2, s_idx, nA)
            combined = [Q1[s_idx][i] + Q2[s_idx][i] for i in range(nA)]
            if random.random() < eps:
                a = random.randrange(nA)
            else:
                maxv = max(combined); choices = [i for i,v in enumerate(combined) if v==maxv]; a = random.choice(choices)
            next_state, r, done, _ = env.step(a)
            ns_idx = env.state_to_index(next_state)
            ensure_Q_entry(Q1, ns_idx, nA); ensure_Q_entry(Q2, ns_idx, nA)
            if random.random() < 0.5:
                best_a = int(np.argmax(Q1[ns_idx]))
                target = r + (0 if done else gamma * Q2[ns_idx][best_a])
                Q1[s_idx][a] += alpha * (target - Q1[s_idx][a])
            else:
                best_a = int(np.argmax(Q2[ns_idx]))
                target = r + (0 if done else gamma * Q1[ns_idx][best_a])
                Q2[s_idx][a] += alpha * (target - Q2[s_idx][a])
            total_r += r
            s_idx = ns_idx
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    # merge
    Q = {}
    keys = set(list(Q1.keys()) + list(Q2.keys()))
    for k in keys:
        v1 = Q1.get(k, [0.0]*nA); v2 = Q2.get(k, [0.0]*nA)
        Q[k] = [v1[i] + v2[i] for i in range(nA)]
    return Q, rewards

def train_monte_carlo(env, episodes=3000, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    nA = 4
    Q = {}
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    rewards = []
    eps = eps_start
    for ep in trange(episodes, desc="MonteCarlo"):
        episode = []
        state = env.reset()
        s_idx = env.state_to_index(state)
        ensure_Q_entry(Q, s_idx, nA)
        done = False
        total_r = 0.0
        while not done:
            a = epsilon_greedy(Q, s_idx, nA, eps)
            next_state, r, done, _ = env.step(a)
            ns_idx = env.state_to_index(next_state)
            episode.append((s_idx, a, r))
            total_r += r
            s_idx = ns_idx
            ensure_Q_entry(Q, s_idx, nA)
        # First-visit updates
        G = 0.0
        visited = set()
        for t in range(len(episode)-1, -1, -1):
            s,a,r = episode[t]
            G = gamma * G + r
            if (s,a) not in visited:
                visited.add((s,a))
                returns_sum[(s,a)] += G
                returns_count[(s,a)] += 1.0
                Q[s][a] = returns_sum[(s,a)] / returns_count[(s,a)]
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    return Q, rewards

# === Policy extraction & plotting helpers (unique keys used) ===
def derive_policy_and_value(Q, env):
    policy = np.zeros((env.H, env.W), dtype=int)
    value = np.zeros((env.H, env.W), dtype=float)
    for r in range(env.H):
        for c in range(env.W):
            s_idx = r*env.W + c
            if s_idx in Q:
                arr = Q[s_idx]
                best_a = int(np.argmax(arr))
                policy[r,c] = best_a
                value[r,c] = max(arr)
    return policy, value

def plot_elevation_heatmap(env, key_suffix="elev"):
    z_display = env.elev[::-1]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z_display, text=np.round(z_display,2), hoverinfo='z', colorbar_title="elev"))
    fig.update_layout(title="Elevation (flipped)", width=600, height=600)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def draw_policy_arrows_on_fig(fig, env, policy, arrow_scale=0.35, line_width=2):
    for r in range(env.H):
        for c in range(env.W):
            a = int(policy[r,c])
            x0 = c; y0 = env.H - 1 - r
            if (r,c) == env.goal: continue
            if a == 0: dx, dy = 0, arrow_scale
            elif a == 1: dx, dy = arrow_scale, 0
            elif a == 2: dx, dy = 0, -arrow_scale
            elif a == 3: dx, dy = -arrow_scale, 0
            else: dx, dy = 0, 0
            x1 = x0 + dx; y1 = y0 + dy
            fig.add_trace(go.Scatter(x=[x0,x1], y=[y0,y1], mode='lines', showlegend=False, line=dict(color='black', width=line_width)))
            # arrowheads
            ang = math.atan2(dy, dx) if (dx!=0 or dy!=0) else 0.0
            head_len = arrow_scale * 0.4
            ang1 = ang + math.radians(135); ang2 = ang - math.radians(135)
            hx1 = x1 + head_len * math.cos(ang1); hy1 = y1 + head_len * math.sin(ang1)
            hx2 = x1 + head_len * math.cos(ang2); hy2 = y1 + head_len * math.sin(ang2)
            fig.add_trace(go.Scatter(x=[x1,hx1], y=[y1,hy1], mode='lines', showlegend=False, line=dict(color='black', width=line_width)))
            fig.add_trace(go.Scatter(x=[x1,hx2], y=[y1,hy2], mode='lines', showlegend=False, line=dict(color='black', width=line_width)))
    return fig

def plot_policy_on_elevation(env, policy, title="Policy on elevation", key_suffix="policy"):
    fig = plot_elevation_heatmap(env, key_suffix=key_suffix)
    draw_policy_arrows_on_fig(fig, env, policy)
    sx, sy = env.start[1], env.H - 1 - env.start[0]
    gx, gy = env.goal[1], env.H - 1 - env.goal[0]
    fig.add_trace(go.Scatter(x=[sx], y=[sy],mode='markers+text',marker=dict(size=14, symbol='circle', line=dict(width=2)),name='start',text=["Start"],textposition="bottom center"))
    fig.add_trace(go.Scatter(x=[gx], y=[gy],mode='markers+text',marker=dict(size=16, symbol='star'),name='goal',text=["Goal"],textposition="bottom center"))
    fig.update_layout(title=title)
    return fig

def plot_value_heatmap(env, value, title="Value heatmap", key_suffix="val"):
    fig = px.imshow(value[::-1], text_auto=True, origin='lower')
    fig.update_layout(title=title, width=700, height=500)
    return fig

def plot_learning_curves(all_rewards, labels, key_suffix="learn"):
    fig = go.Figure()
    for rewards,label in zip(all_rewards, labels):
        w = max(1, len(rewards)//30)
        if len(rewards) >= w:
            smooth = np.convolve(rewards, np.ones(w)/w, mode='same')
        else:
            smooth = rewards
        fig.add_trace(go.Scatter(y=smooth, name=label))
    fig.update_layout(title="Learning Curves (smoothed)", xaxis_title="Episode", yaxis_title="Return", width=900, height=400)
    return fig

def simulate_greedy_trajectory(env, Q, max_steps=500):
    env.reset()
    traj = []
    done = False
    s_idx = env.state_to_index(env._state())
    steps = 0
    while not done and steps < max_steps:
        ensure_Q_entry(Q, s_idx, 4)
        a = int(np.argmax(Q[s_idx]))
        state, r, done, info = env.step(a)
        traj.append((env.pos, info.get("event","")))
        s_idx = env.state_to_index(state)
        steps += 1
    return traj

def plot_trajectory_on_elev(env, traj, title="Trajectory"):
    fig = plot_elevation_heatmap(env)
    xs = [c for (r,c) in [s for s,e in traj]]
    ys = [env.H - 1 - r for (r,c) in [s for s,e in traj]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=4), marker=dict(size=10), name="path"))
    fig.update_layout(title=title, width=600, height=600)
    return fig

# === Streamlit UI ===
st.sidebar.title("Controls")
st.sidebar.markdown("Configure environment, flood, and agent training parameters here.")

# Flood modes include all choices
flood_mode = st.sidebar.selectbox("Flood behavior", options=[
    "static", "global_rise", "bfs_spread", "terrain_bfs", "stochastic", "multi_bfs"
], index=3)

# Environment controls
grid_size = st.sidebar.slider("Grid size N (N x N)", 5, 20, 8)
start_r = st.sidebar.number_input("Start row (0-index)", min_value=0, max_value=grid_size-1, value=grid_size-1)
start_c = st.sidebar.number_input("Start col (0-index)", min_value=0, max_value=grid_size-1, value=0)
goal_r = st.sidebar.number_input("Goal row (0-index)", min_value=0, max_value=grid_size-1, value=0)
goal_c = st.sidebar.number_input("Goal col (0-index)", min_value=0, max_value=grid_size-1, value=grid_size-1)
start = (int(start_r), int(start_c))
goal = (int(goal_r), int(goal_c))

water_rise = st.sidebar.slider("Water rise per step (global)", 0.0, 0.2, 0.02, 0.005)
bfs_base_delay = st.sidebar.slider("BFS base delay (steps)", 1, 4, 1)
terrain_friction = st.sidebar.slider("Terrain friction (slows flood)", 0.0, 2.0, 0.6, 0.05)
random_delay_prob = st.sidebar.slider("Stochastic random delay prob", 0.0, 0.5, 0.15, 0.01)
rainfall_prob = st.sidebar.slider("Rainfall burst probability per step", 0.0, 0.1, 0.02, 0.005)
flood_source_count = st.sidebar.slider("Number of flood source seeds", 1, 5, 1)

# Ensure default max_steps is never below the allowed minimum (50)
max_steps_default = max(4 * grid_size, 50)
max_steps = st.sidebar.number_input(
    "Max steps per episode",
    min_value=50,
    max_value=2000,
    value=max_steps_default,
    step=10
)

seed = st.sidebar.number_input("Random seed (0 for random)", value=42, step=1)
if seed == 0:
    seed = None

# Reward shaping and RL hyperparams
st.sidebar.markdown("---")
st.sidebar.header("Reward & RL hyperparameters")
reward_goal = st.sidebar.number_input("Reward - Goal reached", value=100.0)
reward_drown = st.sidebar.number_input("Reward - Drowned", value=-50.0)
reward_step = st.sidebar.number_input("Reward - Step penalty", value=-1.0)

alpha = st.sidebar.slider("Learning rate (alpha)", 0.0, 1.0, 0.3, 0.01)
gamma = st.sidebar.slider("Discount factor (gamma)", 0.0, 1.0, 0.99, 0.01)
eps_start = st.sidebar.slider("Epsilon start", 0.0, 1.0, 1.0, 0.01)
eps_end = st.sidebar.slider("Epsilon end", 0.0, 1.0, 0.05, 0.01)
eps_decay = st.sidebar.slider("Epsilon decay (per episode)", 0.90, 0.9999, 0.995, 0.0001)

episodes_q = st.sidebar.number_input("Episodes (Q/SARSA/DoubleQ)", min_value=50, max_value=50000, value=1500, step=50)
episodes_mc = st.sidebar.number_input("Episodes (Monte Carlo)", min_value=50, max_value=50000, value=3000, step=50)

# Agent speed for animation / evaluation: number of micro-actions per environment step
agent_speed = st.sidebar.slider("Agent speed (micro-actions per env step during evaluation)", 1, 4, 1)

# Algorithm selection
st.sidebar.markdown("---")
st.sidebar.header("Algorithms")
train_q_flag = st.sidebar.checkbox("Q-Learning", value=True)
train_double_flag = st.sidebar.checkbox("Double-Q", value=True)
train_sarsa_flag = st.sidebar.checkbox("SARSA", value=True)
train_mc_flag = st.sidebar.checkbox("Monte Carlo (First-Visit)", value=False)

# UI behavior
st.sidebar.markdown("---")
train_mode = st.sidebar.radio("Training triggers", options=["Manual (click Train)", "Auto (retrain on parameter change)"], index=0)
st.sidebar.write("Large episode counts can be slow; use Manual during exploration.")

# Buttons
train_btn = st.sidebar.button("Train selected agents")
simulate_btn = st.sidebar.button("Simulate greedy trajectories")
compare_btn = st.sidebar.button("Compare policies (show plots)")
# tune_btn = st.sidebar.button("Quick parameter tuning (small grid)")

# build flood_sources list (randomly choose seeds along bottom row or corners)
def make_flood_sources(count, N):
    seeds = []
    # default seed positions: bottom-left, bottom-right, mid-bottom, top-right, top-left
    candidates = [(N-1,0),(N-1,N-1),(N-1,N//2),(0,N-1),(0,0)]
    for i in range(count):
        seeds.append(candidates[i % len(candidates)])
    return seeds

flood_seeds = make_flood_sources(flood_source_count, grid_size)

# Create environment instance from current UI
env = FloodGridWorld(N=grid_size, start=start, goal=goal, flood_mode=flood_mode, seed=seed,
                     water_rise_per_step=water_rise, bfs_base_delay=bfs_base_delay,
                     terrain_friction=terrain_friction, random_delay_prob=random_delay_prob,
                     rainfall_burst_prob=rainfall_prob, flood_sources=flood_seeds,
                     max_steps=max_steps, reward_goal=reward_goal, reward_drown=reward_drown, reward_step=reward_step)

# Session storage for agents, Qs, rewards
if "agents_Q" not in st.session_state:
    st.session_state.agents_Q = {}
if "agents_rew" not in st.session_state:
    st.session_state.agents_rew = {}
if "last_trained_config" not in st.session_state:
    st.session_state.last_trained_config = None

# helper to detect config changes
def current_config_signature():
    return (grid_size, flood_mode, tuple(flood_seeds), water_rise, bfs_base_delay, terrain_friction,
            random_delay_prob, rainfall_prob, reward_goal, reward_drown, reward_step,
            alpha, gamma, eps_start, eps_end, eps_decay, episodes_q, episodes_mc, seed)

# Auto retrain if selected
if train_mode == "Auto (retrain on parameter change)":
    # if config differs from last trained, retrain automatically
    if st.session_state.last_trained_config != current_config_signature():
        st.info("Auto-training due to parameter change...")
        train_btn = True  # set to emulate pressing train

# Training action
if train_btn:
    st.info("Training selected agents — this may take time depending on episode counts.")
    # clear old ones
    st.session_state.agents_Q = {}
    st.session_state.agents_rew = {}
    st.session_state.last_trained_config = current_config_signature()
    # which algorithms
    algos_to_train = []
    if train_q_flag: algos_to_train.append("Q")
    if train_double_flag: algos_to_train.append("DoubleQ")
    if train_sarsa_flag: algos_to_train.append("SARSA")
    if train_mc_flag: algos_to_train.append("MC")
    if len(algos_to_train) == 0:
        st.warning("No algorithm selected for training.")
    else:
        progress_bar = st.progress(0)
        total = len(algos_to_train)
        i = 0
        for algo in algos_to_train:
            st.write(f"Training {algo} ...")
            if algo == "Q":
                Q, rew = train_q_learning(env, episodes=episodes_q, alpha=alpha, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
            elif algo == "DoubleQ":
                Q, rew = train_double_q(env, episodes=episodes_q, alpha=alpha, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
            elif algo == "SARSA":
                Q, rew = train_sarsa(env, episodes=episodes_q, alpha=alpha, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
            elif algo == "MC":
                Q, rew = train_monte_carlo(env, episodes=episodes_mc, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
            st.session_state.agents_Q[algo] = Q
            st.session_state.agents_rew[algo] = rew
            i += 1
            progress_bar.progress(int(100*i/total))
        st.success("Training finished for selected agents.")

# Show learning curves if present
st.header("Training / Learning Curves")
if len(st.session_state.agents_rew) > 0:
    labels = []
    rews = []
    for k,v in st.session_state.agents_rew.items():
        labels.append(k)
        rews.append(v)
    fig = plot_learning_curves(rews, labels, key_suffix="learning_main")
    st.plotly_chart(fig, use_container_width=True, key="learning_plot")

# Compare policies & values
if compare_btn:
    if len(st.session_state.agents_Q) == 0:
        st.warning("No trained policies to compare. Train agents first.")
    else:
        st.header("Policy visualizations")
        cols = st.columns(2)
        i = 0
        for algo, Qdict in st.session_state.agents_Q.items():
            policy, value = derive_policy_and_value(Qdict, env)
            fig = plot_policy_on_elevation(env, policy, title=f"{algo} policy", key_suffix=f"{algo}_policy")
            fig_val = plot_value_heatmap(env, value, title=f"{algo} value")
            with cols[i%2]:
                st.plotly_chart(fig, use_container_width=True, key=f"{algo}_policy_chart")
                st.plotly_chart(fig_val, use_container_width=True, key=f"{algo}_value_chart")
            i += 1

# Simulate greedy trajectories with animation
if simulate_btn:
    if len(st.session_state.agents_Q) == 0:
        st.warning("No trained agents available. Train first.")
    else:
        st.header("Simulate greedy trajectories (animated)")
        # run each trained agent sequentially
        for algo, Qdict in st.session_state.agents_Q.items():
            st.subheader(f"{algo} greedy trajectory")
            # we need a placeholder per agent with unique key
            placeholder = st.empty()
            # reset environment before simulation
            env.reset()
            s_idx = env.state_to_index(env._state())
            traj = []
            steps = 0
            max_sim_steps = min(500, env.max_steps)
            while steps < max_sim_steps:
                ensure_Q_entry(Qdict, s_idx, 4)
                a = int(np.argmax(Qdict[s_idx]))
                # agent micro-action speed: perform agent_speed micro-steps per environment tick
                micro_steps = agent_speed
                done = False
                for _ in range(micro_steps):
                    state, r, done, info = env.step(a)
                    traj.append((env.pos, info.get("event","")))
                    s_idx = env.state_to_index(state)
                    steps += 1
                    if done:
                        break
                fig = plot_trajectory_on_elev(env, traj, title=f"{algo} greedy (step {steps})")
                # unique key for each frame: algo + frame index
                placeholder.plotly_chart(fig, use_container_width=True, key=f"{algo}_frame_{steps}")
                time.sleep(max(0.05, 0.5/ max(1, agent_speed)))
                if done:
                    st.write("End event:", info.get("event",""))
                    break

# # Quick parameter tuning (small)
# if tune_btn:
#     st.info("Running a quick parameter tuning sweep (small defaults). This will take some time.")
#     # small sweep values for interactivity
#     grid_sizes = [5,6,7,8,9,10,11,12,13,14,15,16]
#     flood_speeds = [0.02, 0.04]
#     agent_speeds = [1, 2]
#     epsilons = [1.0, 0.5]
#     eps_decays = [0.995]
#     alphas = [0.1, 0.3]
#     reward_shapes = [
#         {"goal":100.0, "drown":-50.0, "step":-1.0},
#         {"goal":20.0, "drown":-10.0, "step":-0.1}
#     ]
#     rows = []
#     pbar = st.progress(0)
#     total = len(grid_sizes)*len(flood_speeds)*len(agent_speeds)*len(epsilons)*len(eps_decays)*len(alphas)*len(reward_shapes)*4
#     donecount = 0
#     for N in grid_sizes:
#         for fs in flood_speeds:
#             for aspeed in agent_speeds:
#                 for eps0 in epsilons:
#                     for epsd in eps_decays:
#                         for a_lr in alphas:
#                             for rshape in reward_shapes:
#                                 # create env
#                                 env_local = FloodGridWorld(N, flood_mode=flood_mode, bfs_base_delay=bfs_base_delay,
#                                                            terrain_friction=terrain_friction, random_delay_prob=random_delay_prob,
#                                                            rainfall_burst_prob=rainfall_prob, flood_sources=flood_seeds,
#                                                            max_steps=4*N, reward_goal=rshape["goal"], reward_drown=rshape["drown"], reward_step=rshape["step"],
#                                                            seed=seed)
#                                 for algo in ["Q","SARSA","DoubleQ","MC"]:
#                                     epcount = episodes_mc if algo=="MC" else episodes_q
#                                     # quick training
#                                     if algo == "Q":
#                                         Qlocal, _ = train_q_learning(env_local, episodes=max(200, epcount//4), alpha=a_lr, gamma=gamma, eps_start=eps0, eps_end=eps_end, eps_decay=epsd)
#                                     elif algo == "SARSA":
#                                         Qlocal, _ = train_sarsa(env_local, episodes=max(200, epcount//4), alpha=a_lr, gamma=gamma, eps_start=eps0, eps_end=eps_end, eps_decay=epsd)
#                                     elif algo == "DoubleQ":
#                                         Qlocal, _ = train_double_q(env_local, episodes=max(200, epcount//4), alpha=a_lr, gamma=gamma, eps_start=eps0, eps_end=eps_end, eps_decay=epsd)
#                                     else:
#                                         Qlocal, _ = train_monte_carlo(env_local, episodes=max(300, epcount//3), gamma=gamma, eps_start=eps0, eps_end=eps_end, eps_decay=epsd)
#                                     # evaluate with agent speed micro-actions
#                                     success = drowned = timeout = 0
#                                     eval_episodes = 50
#                                     for _ in range(eval_episodes):
#                                         env_local.reset()
#                                         done=False
#                                         while not done:
#                                             sidx = env_local.state_to_index(env_local._state())
#                                             ensure_Q_entry(Qlocal, sidx, 4)
#                                             a = int(np.argmax(Qlocal[sidx]))
#                                             for _ in range(aspeed):
#                                                 s, r, done, info = env_local.step(a)
#                                                 if done: break
#                                         ev = info.get("event","")
#                                         if ev == "goal": success += 1
#                                         elif ev == "drowned": drowned += 1
#                                         else: timeout += 1
#                                     rows.append({"grid_N":N, "flood_speed":fs, "agent_speed":aspeed, "eps_start":eps0, "eps_decay":epsd, "alpha":a_lr,
#                                                  "reward_goal":rshape["goal"], "algo":algo, "success":success/eval_episodes, "drowned":drowned/eval_episodes})
#                                     donecount += 1
#                                     pbar.progress(int(100*donecount/total))
#     df_tune = pd.DataFrame(rows)
#     st.write("Quick tuning results (sample):")
#     st.dataframe(df_tune.head(20))
#     # bar chart: best success per algo x grid_N
#     st.write("Grouped bar: best success per algorithm and grid size (from quick tuning)")
#     bests = df_tune.sort_values("success", ascending=False).groupby(["algo","grid_N"]).first().reset_index()
#     pivot = bests.pivot(index="grid_N", columns="algo", values="success").fillna(0)
#     fig = px.bar(pivot, barmode="group", title="Best success per algorithm vs grid size (quick tuning)")
#     st.plotly_chart(fig, use_container_width=True)

#     # auto conclusions
#     st.write("Auto conclusions (quick):")
#     concl = []
#     for algo in pivot.columns:
#         best_idx = pivot[algo].idxmax()
#         best_val = pivot[algo].max()
#         concl.append(f"{algo}: best success {best_val:.2f} at grid {best_idx}")
#     st.write("\n".join(concl))

# === Download code button ===
st.sidebar.markdown("---")
if st.sidebar.button("Download app.py"):
    # Return this file's code to user
    import inspect, sys
    code = inspect.getsource(sys.modules[__name__])
    st.sidebar.download_button("Download app.py", data=code, file_name="app.py")


st.markdown("---")
st.caption("App created for Flood Escape RL comparisons.")
