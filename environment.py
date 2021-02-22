import random
import torch
from unityagents import UnityEnvironment

class Environment():
    """
    A class wrapper for the unity environment with some helper functions.
    By default initialises the "Tennis" environment but could initialise other environments.
    Has helper methods for manipulating the environment: 
    step(), reset(), close()
    Has helper methods for returning various data points which explain the environment:
    get_action_size(), get_state_size(), get_brain_name(), get_num_of_agents(), get_states_per_agent()
    """
    def __init__(self, file_name="Tennis.app", seed=0, train_mode=False, no_graphics=False):
        # Initialise random seed environment:
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics, seed=seed)
        self.train_mode = train_mode
        
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        
        self.action_size = self.brain.vector_action_space_size
        self.num_of_agents = len(self.env_info.agents)
        
        self.states = self.env_info.vector_observations
        self.state_size = self.states.shape[1]
        self.states_per_agent = self.state_size * self.num_of_agents
        
        self.print_env_info()
        
    def get_action_size(self):
        return self.action_size
    
    def get_state_size(self):
        return self.state_size
        
    def get_brain_name(self):
        return self.brain_name
    
    def get_num_of_agents(self):
        return self.num_of_agents
    
    def get_states_per_agent(self):
        return self.states_per_agent
        
    def print_env_info(self):
        print("Brain name: {}\n".format(self.brain_name))
        print("Number of agents: {}\n".format(self.num_of_agents))
        print("Each agent observes a state with length: {}\n".format(self.state_size))
        print("Each agent will receive a state size of: {}\n".format(self.states_per_agent))
        print("The state for the first agent looks like: {}\n".format(self.states))
        
    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name]
        self.states = self.env_info.vector_observations
    
    def reset(self):
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self.states = self.env_info.vector_observations
        
    def close(self):
        self.env.close()