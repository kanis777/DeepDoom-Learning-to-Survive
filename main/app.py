from flask import Flask, render_template, request
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time
from flask import redirect, url_for
from threading import Thread
import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global reference to the active environment
active_env = None

app = Flask(__name__)

# Absolute paths for models
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_PATHS = {
    "basic": {
        "ppo": os.path.join(BASE_PATH, "models", "basic_ppo_1.zip"),
        "hardcoded_ppo": os.path.join(BASE_PATH, "models", "ppo_vizdoom_best_2.pt"),
        "a2c": os.path.join(BASE_PATH, "models", "basic_a2c.zip"),
        "dqn": os.path.join(BASE_PATH, "models", "basic_dqn.zip")
    },
    "defend": {
        "ppo": os.path.join(BASE_PATH, "models", "defend_ppo_1.zip")
    },
    "deadly": {
        "ppo": os.path.join(BASE_PATH, "models", "deadly_ppo_1.zip")
    }
}


def load_env(scenario, render):
    if scenario == "basic":
        from basic_env import VizDoomGymBasic
        return VizDoomGymBasic(render)
    elif scenario == "defend":
        from defend_env import VizDoomGymDefend
        return VizDoomGymDefend(render)
    elif scenario == "deadly":
        from deadly_env import VizDoomGymDeadly
        return VizDoomGymDeadly(render)
    else:
        raise ValueError("Invalid scenario")

def evaluate_hardcoded(model_path,scenario, episodes=5, render=True):
    env = load_env(scenario, render=True)
    from basic_hardcoded import ActorCritic
    model = ActorCritic(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    episode_rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = model(state_tensor)
                action = torch.argmax(logits, dim=-1).item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                cv2.imshow("ViZDoom", np.squeeze(state))
                cv2.waitKey(10)
        print(f"Episode {ep + 1} finished with total reward: {total_reward}")
        episode_rewards.append(round(total_reward, 2))
    env.close()
    cv2.destroyAllWindows()
    return np.mean(episode_rewards).round(2), episode_rewards

def run_evaluation(model_path, scenario, result_holder):
    global active_env
    print(f"Evaluating model: {model_path}")

    if not os.path.exists(model_path):
        result_holder["result"] = (f"Model not found at path: {model_path}", [])
        return

    env = load_env(scenario, render=True)
    active_env = env  # Save active env globally

    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        result_holder["result"] = ("Unsupported model type", [])
        return

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    episode_rewards = []
    for ep in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(0.20)
            total_reward += reward
        episode_rewards.append(round(total_reward, 2))

    result_holder["result"] = (round(mean_reward, 2), episode_rewards)
    env.close()
    active_env = None
    
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        scenario = request.form.get("scenario")
        agent = request.form.get("agent")

        if scenario in MODEL_PATHS and agent in MODEL_PATHS[scenario]:
            model_path = MODEL_PATHS[scenario][agent]
            result_holder = {}

            if agent == "hardcoded_ppo":
                # Handle hardcoded PPO separately
                thread = Thread(target=run_hardcoded_evaluation, args=(model_path, scenario, result_holder))
            else:
                # Regular evaluation for PPO, A2C, DQN, etc.
                thread = Thread(target=run_evaluation, args=(model_path, scenario, result_holder))
            
            thread.start()
            thread.join()  # Wait until evaluation is done
            result = result_holder.get("result")
        else:
            result = ("Invalid agent or scenario selected.", [])

    return render_template("index.html", result=result)

def run_hardcoded_evaluation(model_path, scenario, result_holder):
    print(f"Evaluating hardcoded PPO model: {model_path}")

    if not os.path.exists(model_path):
        result_holder["result"] = (f"Model not found at path: {model_path}", [])
        return

    mean_reward, episode_rewards = evaluate_hardcoded(model_path, scenario)

    result_holder["result"] = (mean_reward, episode_rewards)


@app.route("/stop", methods=["POST"])
def stop_env():
    global active_env
    if active_env:
        print("Closing active environment...")
        active_env.close()
        active_env = None
    return redirect(url_for("index"))

if __name__ == "__main__":
    print("Flask server is starting:")
    app.run(debug=True)
