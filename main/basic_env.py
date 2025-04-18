#define vizdppm openai environment class
from vizdoom import *
from vizdoom import DoomGame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2

class VizDoomGymBasic(Env):
    def __init__(self,render=False): #wt we need for starting our game
        super().__init__()#to inherit from base class Env
        self.game=DoomGame()
        self.game.load_config('D:/sem8/RL_lab/package/ViZDoom/scenarios/basic.cfg')
        
        if render==False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        self.game.init() 
        #self.observation_space=Box(low=0,high=255,shape=(240,320,3),dtype=np.uint8)
        self.observation_space=Box(low=0,high=255,shape=(100,160,1),dtype=np.uint8)
        self.action_space=Discrete(3)
        
    def step(self,action): #to take a specific action and take step in the environment
        actions=np.identity(3,dtype=np.uint8)
        reward=self.game.make_action(actions[action],2)
        if self.game.get_state():
            state=self.game.get_state()
            img=state.screen_buffer
            img=self.grayscale(img)
            ammo=state.game_variables[0]
            info={"info":ammo}
        else:
            img=np.zeros(self.observation_space.shape)
            val=0
            info={"info":val}
        done=self.game.is_episode_finished()
        return img,reward,done,info
    def render(self):#define how to render the game
       pass
    
    def reset(self):#when we start a new game
        self.game.new_episode()
        state=self.game.get_state()
        img=state.screen_buffer
        return self.grayscale(img)
    def grayscale(self,observation):
        gray=cv2.cvtColor(np.moveaxis(observation,0,-1),cv2.COLOR_BGR2GRAY)#to make it 240,320,3
        resize=cv2.resize(gray,(160,100),interpolation=cv2.INTER_CUBIC)
        state=np.reshape(resize,(100,160,1))
        return state
    def close(self):#to close the game
        self.game.close()