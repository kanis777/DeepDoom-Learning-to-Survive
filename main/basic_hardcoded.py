from vizdoom import *
from vizdoom import DoomGame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2
import torch.nn as nn
 
class VizDoomGym(Env):
    def __init__(self, render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config('ViZDoom/scenarios/basic.cfg')
        self.game.set_window_visible(render)
        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 2)
        done = self.game.is_episode_finished()
        if done:
            state = np.zeros(self.observation_space.shape)
            info = {"ammo": 0}
        else:
            state = self.grayscale(self.game.get_state().screen_buffer)
            info = {"ammo": self.game.get_state().game_variables[0]}
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.grayscale(self.game.get_state().screen_buffer)
        return state

    def render(self): pass

    def grayscale(self, obs):
        gray = cv2.cvtColor(np.moveaxis(obs, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (100, 160, 1))

    def close(self):
        self.game.close()

# 2. Define Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 16, 512), nn.ReLU()
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)