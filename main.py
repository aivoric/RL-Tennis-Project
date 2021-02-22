from box import Box
from trainer import Trainer
from tester import Tester

"""
=============================================================================
STEP 1: Create the main function
=============================================================================
main() is the entry point into the program.
It initiliases one of the following clases depending on the mode:
1) Trainer -> used for training agents
2) Tester -> used for walking trained agent(s) through an environment
"""
def main(environment_file_name, hyperparameters, random_seed, train_mode=False):
    if train_mode:
        trainer = Trainer(environment_file_name, hyperparameters, random_seed=random_seed)
        trainer.train()
        trainer.display_final_result()
    else:
        tester = Tester(environment_file_name, hyperparameters, random_seed=random_seed, games_to_play=10)
        tester.load_weights()
        tester.play()


"""
=============================================================================
STEP 2: Initialise hyperparameters and other control variables.
=============================================================================
The Box() class allows for the hyperparameter dictionary to be referenced 
safely like this: hyperparameters.batch_size, anywhere it is used.

This hyperparameter dictionary is passed to all the classes used in the program
so additional variables can easily be inserted here, and used across the program.

To safely save results separately, increment the value of the ITERATION. 
That will save results in a new folder with the corresponding number,
i.e. results/results_i 
"""

hyperparameters = Box({
    'ITERATION': 8,               # Iteration ID allows to put results into new folders
    'EPISODES': 5000,             # Number of episode to loop through
    'SAVE_EVERY': 100,            # How often to save the agent
    'BUFFER_SIZE': int(1e5),      # Replay buffer size
    'BATCH_SIZE': 512,            # Training batch size
    'GAMMA': 0.99,                # Discount factor
    'TAU': 0.15,                  # Soft update multiplier
    'LR_ACTOR': 0.00005,          # Learning rate of the actor 
    'LR_CRITIC': 0.0003,          # Learning rate of the critic
    'WEIGHT_DECAY': 0.0000,       # L2 weight decay
    'MU': 0.0,                    # Ornstein-Uhlenbeck config -> the starting state of Ornstein-Uhlenbeck process
    'THETA': 0.15,                # Ornstein-Uhlenbeck config
    'SIGMA': 0.2,                 # Ornstein-Uhlenbeck config -> sigma multiplier
    'USE_SIGMA_DECAY': False,     # Set to True if you want Sigma to decay over time. Then control the decay with min and decay values.
    'SIGMA_MIN': 0.05,            # Ornstein-Uhlenbeck config -> minimum value to which to decay to. 
    'SIGMA_DECAY': 0.99           # Ornstein-Uhlenbeck config -> decay multiplier to reduce sigma
})
    
# Choose an environment
environment_file_name="Tennis.app"
#environment_file_name="Soccer.app"

"""
=============================================================================
STEP 3: Define train/test mode, set a seed, and launch and program
=============================================================================
'train_mode = True' will train agent(s) in the above selected environment
'random_seed' allows for consistent random number generation to help
reproduce results.
'random_seed' is passed to all the classes used in the program:
Environment, Agent, OUNoise, ReplayBuffer
"""
train_mode = True
random_seed=0
main(environment_file_name, hyperparameters, random_seed, train_mode)