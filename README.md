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

### `basic.cfg`
The purpose of the scenario is just to check if using this framework to train some Al in a 3D environment is feasible. 
Map is a rectangle with gray walls, ceiling and floor. Player is spawned along the longer wall, in the center. A red, circular monster is spawned randomly somewhere along the opposite wall. Player can only (config) go left/right and shoot. 1 hit is enough to kill the monster. Episode finishes when monster is killed or on timeout. 
REWARDS: 
+101 for killing the monster -5 for missing Episode ends after killing the monster or on timeout. 
Further configuration: 
living reward = -1, 
3 available buttons: move left, move right, shoot (attack) 
timeout = 300

### `defend_the_center.cfg`
The purpose of this scenario is to teach the agent that killing the monsters is GOOD and when monsters kill you is BAD. In addition, wasting amunition is not very good either. Agent is rewarded only for killing monsters so he has to figure out the rest for himself. 
Map is a large circle. Player is spawned in the exact center. 5 melee-only, monsters are spawned along the wall. Monsters are killed after a single shot. After dying each monster is respawned after some time. Episode ends when the player dies (it's inevitable becuse of limitted ammo). 
REWARDS: +1 for killing a monster 
Further configuration: 
3 available buttons: turn left, turn right, shoot (attack) 
death penalty = 1

### `deadly_corridor.cfg` (Curriculum Learning)
The purpose of this scenario is to teach the agent to navigate towards his fundamental goal (the vest) and make sure he survives at the same time. 
Map is a corridor with shooting monsters on both sides (6 monsters in total). A green vest is placed at the oposite end of the corridor. Reward is proportional (negative or positive) to change of the distance between the player and the vest. If player ignores monsters on the sides and runs straight for the vest he will be killed somewhere along the way. To ensure this behavior doom_skill = 5 (config) is needed. 
REWARDS: 
+dX for getting closer to the vest. -dX for getting further from the vest. 
Further configuration: 
5 available buttons: turn left, turn right, move left, move right, shoot (attack) 
timeout = 4200 
death penalty = 100 
doom_skill = 5
- Curriculum: Trained in stages from `s1` to `s5` with increasing difficulty

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

DeepDoom/ │ ├── main/ │ ├── app.py │ ├── basic_env.py │ ├── basic_hardcode.py │ ├── defend_env.py │ ├── deadly_env.py │ ├── models/ # Trained agents │ ├── templates/ │ │ └── index.html # UI Page │ └── static/ │ └── style.css # Web styles │ ├── deep_doom.ipynb # Training Notebook ├── requirements.txt └── README.md

yaml
Copy
Edit

---

## 🛠️ How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/DeepDoom.git
   cd DeepDoom
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Launch Flask App

bash
Copy
Edit
cd main
python app.py
Explore Notebooks

deep_doom.ipynb contains training logic and visualization.
