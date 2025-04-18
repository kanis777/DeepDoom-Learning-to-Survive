# 🧠💥 DeepDoom: Reinforcement Learning with ViZDoom

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
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
- Simple left/right movement and shooting
- Kill the monster or timeout
- Rewards:  
  - +101 for kill  
  - -5 for miss  
  - -1 living penalty  

### `defend_the_center.cfg`
- Survive in the center against melee monsters
- Rewards:  
  - +1 for monster kill  
  - -1 for death  

### `deadly_corridor.cfg` (Curriculum Learning)
- Navigate a monster-filled corridor to get a vest
- Rewards:  
  - +dX for moving toward the vest  
  - -dX for moving away  
  - -100 for death  
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
- Upload and select trained models
- Visualize gameplay
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
