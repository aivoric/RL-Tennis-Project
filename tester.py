import numpy as np
import torch
from agent import DDPGAgent
from environment import Environment

class Tester():
    """
    Tester handles walking a trained agent through the environment and showing it visually.
    It initialises its own copies of the environment and agents.
    """
    def __init__(self, environment_file_name, hyperparameters, games_to_play):
        self.env = Environment(file_name=environment_file_name, train_mode=False, no_graphics=False)
        
        # Create the agents which will play against each other:
        self.agent1 = DDPGAgent(state_size=self.env.get_states_per_agent(),
                                action_size=self.env.get_action_size(),
                                hyperparameters=hyperparameters,
                                num_agents=1)
        self.agent2 = DDPGAgent(state_size=self.env.get_states_per_agent(),
                                action_size=self.env.get_action_size(),
                                hyperparameters=hyperparameters,
                                num_agents=1)
        self.games_to_play = games_to_play
    
    def load_weights(self):
        """
        Load trained model weights into the initialised agents.
        """
        #TODO: Move the weights file selection to main.py
        self.agent1.actor_local.load_state_dict(torch.load('results/results_2/agent1_actor.pth'))
        self.agent1.critic_local.load_state_dict(torch.load('results/results_2/agent1_critic.pth'))

        self.agent2.actor_local.load_state_dict(torch.load('results/results_2/agent2_actor.pth'))
        self.agent2.critic_local.load_state_dict(torch.load('results/results_2/agent2_critic.pth'))
    
    def play(self):
        """
        Walk trained agents through a pre-defined number of episodes.
        """
        for episode in range(1, self.games_to_play + 1):

            # Reset environment, agents, and scores
            self.env.reset()
            self.agent1.reset()
            self.agent2.reset()
            episode_scores = np.zeros(self.env.get_num_of_agents())  # each agent will keep track of their own scores

            # Get initial state of the unity environment and reshape it
            states = np.reshape(self.env.states, (1, self.env.get_states_per_agent())) # reshape so we can feed both agents states to each agent
        
            
            while True:
                # Get the first agent actions based on current state, using noise for exploration:
                actions1 = self.agent1.act(states, add_noise=False)
                actions2 = self.agent2.act(states, add_noise=False)

                # Concatenate actions and reshape the data. Then send these actions to the environment:
                actions = np.concatenate((actions1, actions2), axis=0) 
                actions = np.reshape(actions, (1, 4))
                self.env.step(actions)

                # Get a response from the environment:
                next_states = self.env.states      
                next_states = np.reshape(next_states, (1, self.env.get_states_per_agent()))
                rewards = self.env.env_info.rewards                      
                dones = self.env.env_info.local_done                     

                # Set new states to current states so that the next actions can be determined:
                states = next_states
        
                # Update episode score for each agent:
                episode_scores += rewards

                # If one of the agents is done, then exit the loop:
                if np.any(dones):
                    break
        
        # Close environment after game has finished.
        self.env.close()