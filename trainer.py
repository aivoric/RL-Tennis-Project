import numpy as np
import torch
import pickle
from collections import deque
from pathlib import Path
from os import makedirs, path
from time import time
from agent import DDPGAgent
from environment import Environment

class Trainer():
    """
    Trainer handles the entire training process of the agents.
    It imports and creates an environment and the necessary agents.
    It also keeps track of various scores through attributes.
    It has several important methods:
    - train() -> steps through the environment and trains the agents
    - process_scores() -> updates scores, prints them
    - save_models() -> saves the models' weights
    - save_scores() -> saves the scores in a pickle
    - display_final_result() -> displays final results
    """
    def __init__(self, environment_file_name, hyperparameters):
        self.env = Environment(file_name=environment_file_name, train_mode=True, no_graphics=True)
        self.hyperparameters = hyperparameters
        
        #TODO: Finish converting agents into a dynamic list, so that it works in the Soccer environment:
        self.agents = []
        for i in range(self.env.get_num_of_agents()):
            self.agents.append(DDPGAgent(state_size=self.env.get_states_per_agent(),
                                action_size=self.env.get_action_size(),
                                hyperparameters=hyperparameters,
                                num_agents=1
                                ))
        
        # Create the agents which will play against each other:
        self.agent1 = DDPGAgent(state_size=self.env.get_states_per_agent(),
                                action_size=self.env.get_action_size(),
                                hyperparameters=hyperparameters,
                                num_agents=1)
        self.agent2 = DDPGAgent(state_size=self.env.get_states_per_agent(),
                                action_size=self.env.get_action_size(),
                                hyperparameters=hyperparameters,
                                num_agents=1)
        
        # Create various attributes to keep track of scores and other information
        self.episodes = hyperparameters.EPISODES
        self.save_every = hyperparameters.SAVE_EVERY
        self.iteration = hyperparameters.ITERATION
        self.current_episode = 0
        self.scores_window = deque(maxlen=100)
        self.scores = []
        self.average_scores = []
        self.best_score = 0
        self.solved = False
        self.solved_after = 0
        self.solved_result = 0
    
    def train(self):
        """
        The main training loop which goes through each episode, and then steps
        through the environment within each episode.
        At the end of each episode process_scores() is called to update
        the scores attributes and save the models.
        """
        for episode in range(1, self.episodes+1):

            # Reset environment, agents, and scores
            self.env.reset()
            for agent in self.agents:
                agent.reset()
            self.agent1.reset()
            self.agent2.reset()
            episode_scores = np.zeros(self.env.get_num_of_agents())

            # Get initial state of the unity environment and reshape it
            states = np.reshape(self.env.states, (1, self.env.get_states_per_agent()))

            #########################################################################
            #------------------------------------------------------------------------
            # Within the episode step through the environment. The following happens:
            # 1) agents obtain their actions based on current state
            # 2) the environment receives the agents actions'
            # 3) the environment retuns the next state, rewards, dones
            # 4) the agents are updated
            # 5) scores are updated
            while True:
                # Get the first agent actions based on current state, using noise for exploration:
                #TODO: Finish converting agents into a dynamic list, so that it works in the Soccer environment:
                actions1 = self.agent1.act(states, add_noise=True)
                actions2 = self.agent2.act(states, add_noise=True)

                # Concatenate actions and reshape the data. Then send these actions to the environment:
                #TODO: Finish converting agents into a dynamic list, so that it works in the Soccer environment:
                actions = np.concatenate((actions1, actions2), axis=0) 
                actions = np.reshape(actions, (1, 4))
                self.env.step(actions)

                # Get a response from the environment:
                next_states = self.env.states
                next_states = np.reshape(next_states, (1, self.env.get_states_per_agent()))
                rewards = self.env.env_info.rewards                      
                dones = self.env.env_info.local_done                     

                # Save the (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
                #TODO: Finish converting agents into a dynamic list, so that it works in the Soccer environment:
                self.agent1.step(states, actions1, rewards[0], next_states, dones[0])
                self.agent2.step(states, actions2, rewards[1], next_states, dones[1])

                # Set new states to current states so that the next actions can be determined:
                states = next_states
        
                # Update episode score for each agent:
                episode_scores += rewards

                # If one of the agents is done, then exit the loop:
                if np.any(dones):
                    break
            
            # Update current episode and process the scores:
            self.current_episode+= 1
            self.process_scores(episode_scores)

        # Close environment after training is done
        self.env.close()
            
    def process_scores(self, episode_scores):
        """
        Receives the scores from an episode and then:
        1) updates the scores attributes to keep track of them
        2) if a better score across 100 episodes has been found, then save the model weights
        3) print results
        4) if the environment was solved, then display that, and save solved model weights
        """
        # Get the max score from the agents:
        max_score = np.max(episode_scores) 
        
        # Update the score trackers:
        self.scores_window.append(max_score) 
        self.scores.append(max_score)
        
        # Calculate the average score:
        average_score = np.mean(self.scores_window)
        
        # Update the average score tracker
        self.average_scores.append(average_score)
        
        # If average score is better than the best score, then update best score and save model weights:
        if average_score > self.best_score: 
            self.best_score = average_score
            self.save_models()
        
        # Print result of every episode. end="" ensures that result is printed on the same line.
        print('\rEpisode: {} \t Episode score: {:.2f} \t Average Score: {:.2f}'.format(
            self.current_episode, max_score, average_score), end='')
        
        # Print checkpoint results and save scores:
        if self.current_episode % self.save_every == 0:
            print('\r## CHECKPOINT ## Episode: {}   Best score: {:.2f}   Average Score Last 100 Episodes: {:.2f}'.format(
                self.current_episode, self.best_score, average_score))
            self.save_scores()
        
        # If episode resulted in the environment being solved:
        if average_score >= 0.5 and not self.solved:
            self.solved = True
            print('\r## CONGRATS! ## Environment solved after {} episodes. Average Score Last 100 Episodes: {:.2f}'.format(
                self.current_episode, average_score))
            self.save_models(solved=True)
    
    def save_models(self, solved=False):
        """
        1. Create a folder if it doesn't exist where to store results.
        2. Save Actor and Critic model weights.
        3. Save Actor and Critic model weights for a solved environment (only once)
        """
        # Define the results save location for the current iteration
        results_folder = path.join("results", "results_{}".format(self.iteration))
        
        # Try to create the folder if it doesn't exist
        try:
            makedirs(results_folder)
        except FileExistsError:
            pass
        
        # Define the save location for the model weights:
        agent1_actor = path.join(results_folder, 'agent1_actor.pth'.format())
        agent1_critic = path.join(results_folder, 'agent1_critic.pth'.format())
        agent2_actor = path.join(results_folder, 'agent2_actor.pth'.format())
        agent2_critic = path.join(results_folder, 'agent2_critic.pth'.format())
        
        # Save the latest model weights:
        torch.save(self.agent1.actor_local.state_dict(), agent1_actor)
        torch.save(self.agent1.critic_local.state_dict(), agent1_critic)
        torch.save(self.agent2.actor_local.state_dict(), agent2_actor)
        torch.save(self.agent2.critic_local.state_dict(), agent2_critic)

        # Save the model weights for the checkpoint when the environment was solved:
        if solved:
            # Update the class with how long it took to solve:
            self.solved_after = self.current_episode
            self.solved_result = self.best_score
            
            agent1_actor = path.join(results_folder, 'solved_agent1_actor.pth'.format())
            agent1_critic = path.join(results_folder, 'solved_agent1_critic.pth'.format())
            agent2_actor = path.join(results_folder, 'solved_agent2_actor.pth'.format())
            agent2_critic = path.join(results_folder, 'solved_agent2_critic.pth'.format())
            torch.save(self.agent1.actor_local.state_dict(), agent1_actor)
            torch.save(self.agent1.critic_local.state_dict(), agent1_critic)
            torch.save(self.agent2.actor_local.state_dict(), agent2_actor)
            torch.save(self.agent2.critic_local.state_dict(), agent2_critic)
                
    def save_scores(self):
        """
        Store scores in a pickle
        """
        results_dict = {
            'iteration': self.iteration,
            'episodes': self.episodes,
            'timestamp': int(time()),
            'best_score': self.best_score,
            'all_scores': self.scores,
            'average_scores': self.average_scores
        }
        # Define the results save location for the current iteration
        results_folder = path.join("results", "results_{}".format(self.iteration))
        
        results_save_location = path.join(results_folder, 'scores.pickle'.format())
        
        with open(results_save_location, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def display_final_result(self):
        """
        Display the final results.
        """
        print("\nThe algorithm looped through {} episodes and achieved the best score of {}.".format(
            self.current_episode, self.best_score
        ))
        if self.solved:
            print("\nCongrats! The agent solved the environment in {} episodes with a score of {}.".format(
                self.solved_after, self.solved_result
            ))
            print('')