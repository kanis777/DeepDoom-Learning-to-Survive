from vizdoom import *
from vizdoom import DoomGame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2

class VizDoomGymDeadly(Env):
    def __init__(self,render=False,config='D:/sem8/RL_lab/package/ViZDoom/scenarios/deadly_corridor_s1.cfg'): 
        super().__init__()#to inherit from base class Env
        self.game=DoomGame()
        self.game.load_config(config)
        
        if render==False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        self.game.init() 
        #self.observation_space=Box(low=0,high=255,shape=(240,320,3),dtype=np.uint8)
        self.observation_space=Box(low=0,high=255,shape=(100,160,1),dtype=np.uint8)
        self.action_space=Discrete(7)
        
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 50
        
    def step(self,action): #to take a specific action and take step in the environment
        actions=np.identity(7,dtype=np.uint8)
        movement_reward = self.game.make_action(actions[action], 2) 
        reward = 0 
        if self.game.get_state():
            state=self.game.get_state()
            img=state.screen_buffer
            img=self.grayscale(img)
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables
            
            # change in damage from previous frames
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            reward = movement_reward + damage_taken_delta*10 + hitcount_delta*200  + ammo_delta*5 
            info ={"info":ammo}
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
