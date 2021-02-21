from box import Box
from trainer import Trainer
from tester import Tester

"""
main() is the entry point into the program.
It initiliases one of the following cases depending on the mode:
1) Trainer -> used for training agents
2) Tester -> used for walking trained agent(s) through an environments
"""
def main(environment_file_name, hyperparameters, train_mode=False):
    if train_mode:
        trainer = Trainer(environment_file_name, hyperparameters)
        trainer.train()
        trainer.display_final_result()
    else:
        tester = Tester(environment_file_name, hyperparameters, games_to_play=10)
        tester.load_weights()
        tester.play()


"""
Initialise all the hyperparameters as well as several control variables.
The Box() class allows for the hyperparameter dictionary to be referenced 
safely like this: hyperparameters.batch_size, anywhere it is used
"""
hyperparameters = Box({
    'ITERATION': 5,               # Iteration ID allows to put results into new folders
    'EPISODES': 5000,             # Number of episode to loop through
    'SAVE_EVERY': 100,            # How often to save the agent
    'BUFFER_SIZE': int(5e5),      # Replay buffer size
    'BATCH_SIZE': 128,            # Training batch size
    'GAMMA': 0.99,                # Discount factor
    'TAU': 0.005,                 # Soft update multiplier
    'LR_ACTOR': 1e-3,             # Learning rate of the actor 
    'LR_CRITIC': 1e-3,            # Learning rate of the critic
    'WEIGHT_DECAY': 0             # L2 weight decay
})
    
# Choose an environment
environment_file_name="Tennis.app"
#environment_file_name="Soccer.app"

"""
Launch the program
train_mode = True will train agent(s) in the above selected environment
To safely save results separately, increment the value of the ITERATION
parameter inside the hyperparameter. That will save results in a new folder
with the corresponding number.
"""
train_mode = True
main(environment_file_name, hyperparameters, train_mode)