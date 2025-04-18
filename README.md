# 🧠💥 DeepDoom: Reinforcement Learning with ViZDoom

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.4-blue?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-red?logo=pytorch">
  <img src="https://img.shields.io/badge/Gym-0.26.2-brightgreen?logo=openai">
  <img src="https://img.shields.io/badge/Stable--Baselines3-2.2.1-blueviolet?logo=github">
  <img src="https://img.shields.io/badge/ViZDoom-1.2.0-lightgrey?logo=doom">
  <img src="https://img.shields.io/badge/WandB-Logging-yellow?logo=weightsandbiases">
</p>

## 🎯 Project Overview

**DeepDoom** is a deep reinforcement learning project built on top of [ViZDoom](https://github.com/mwydmuch/ViZDoom), a Doom-based RL platform. This project explores the training and evaluation of AI agents using **PPO**, **A2C**, and **DQN** algorithms in complex FPS game scenarios.

The goal is to teach agents how to navigate, aim, survive, and eliminate enemies across different challenge levels in Doom.

> 🔧 Built with custom Gym-compatible environments, visualization, model evaluation, curriculum learning, and a Flask web UI for real-time interaction with trained agents.

---

## 🧠 RL Algorithms Implemented

| Algorithm | Type           | Policy Type | Notes |
|----------|----------------|-------------|-------|
| **PPO**  | On-policy      | Stochastic  | Proximal Policy Optimization |
| **A2C**  | On-policy      | Stochastic  | Advantage Actor-Critic |
| **DQN**  | Off-policy     | Deterministic | Deep Q-Network |

---

## 🗺️ Scenarios Implemented

## 🧠 Problem Statements / Scenarios

### 🟩 `basic.cfg`
> The purpose of the scenario is just to check if using this framework to train some AI in a 3D environment is feasible.  
> Map is a rectangle with gray walls, ceiling and floor. The player is spawned along the longer wall, in the center. A red, circular monster is spawned randomly somewhere along the opposite wall.  
> Player can only move left, move right, or shoot. One hit is enough to kill the monster. The episode finishes when the monster is killed or times out.

- **Rewards:**
  - 🟥 +101 for killing the monster  
  - ❌ -5 for missing a shot  
  - ⏳ Living reward = -1 per step  
- **Buttons:** Move Left, Move Right, Shoot (3 actions)  
- **Timeout:** 300 steps

---

### 🟨 `defend_the_center.cfg`
> The purpose of this scenario is to teach the agent that killing monsters is GOOD, and getting killed by monsters is BAD.  
> Wasting ammunition is also discouraged.  
> The agent is only rewarded for killing monsters — it must learn to survive and manage its ammo.  
> The player is spawned at the center of a circular map, surrounded by melee-only monsters that respawn after death.  
> The episode ends when the player dies (inevitable due to limited ammo).

- **Rewards:**
  - 🔫 +1 for killing a monster  
  - 💀 Death penalty = -1  
- **Buttons:** Turn Left, Turn Right, Shoot (3 actions)

---

### 🟥 `deadly_corridor.cfg` (⚠️ Curriculum Learning)
> The purpose of this scenario is to teach the agent to **navigate toward its goal (a green vest)** while surviving.  
> The map is a corridor filled with **6 shooting monsters**.  
> The player is rewarded for getting closer to the vest and penalized for going backward.  
> If the player ignores monsters and charges ahead, it will likely be killed before reaching the goal.  
> To make survival necessary, the difficulty level is set using `doom_skill = 5`.

- **Rewards:**
  - ➕ +dX for getting closer to the vest  
  - ➖ -dX for moving away from the vest  
  - 💀 Death penalty = -100  
- **Buttons:** Turn Left, Turn Right, Move Left, Move Right, Shoot (5 actions)  
- **Timeout:** 4200 steps  
- **Difficulty:** `doom_skill = 5`

#### 📈 Curriculum Learning Strategy:
Trained in **5 stages** using:
- `deadly_corridor_s1.cfg`
- `deadly_corridor_s2.cfg`
- `deadly_corridor_s3.cfg`
- `deadly_corridor_s4.cfg`
- `deadly_corridor_s5.cfg`  
Each level increases difficulty. Training continues from one stage’s model to the next with total **200,000 steps**.

---

## 🧪 Training Overview

| Scenario           | Algorithms Used         | Training Steps |
|--------------------|-------------------------|----------------|
| basic.cfg          | PPO, A2C, DQN, Hardcoded PPO | 60,000         |
| defend_the_center  | PPO                     | 100,000        |
| deadly_corridor    | PPO (Curriculum Learning)| 200,000 total  |

**Training was visualized with WandB and evaluated using SB3’s `evaluate_policy()` utility.**

---

## 🌐 Flask Web UI

A simple Flask web interface allows you to:
- Visualize how each model works after the training 
- Interact with agents in real time

### 🌲 Folder Structure

DeepDoom/
├── main/
│   ├── app.py
│   ├── basic_env.py
│   ├── basic_hardcode.py
│   ├── defend_env.py
│   ├── deadly_env.py
│   ├── models/
│   │   ├── basic_ppo_1.zip
│   │   ├── deadly_ppo_1.zip
│   │   ├── defend_ppo_1.zip
│   │   ├── basic_dqn.zip
│   │   ├── basic_a2c.zip
│   │   └── ppo_vizdoom_best.ppt
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── deep_doom.ipynb
├── requirements.txt
└── README.md

---

## 🛠️ How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/DeepDoom.git
   cd DeepDoom```
2. **Install dependencies**

```bash
pip install -r requirements.txt```

3. **Launch Flask App**

```bash
cd main
python app.py```

4. **Explore Notebooks**

deep_doom.ipynb contains training logic and visualization.
