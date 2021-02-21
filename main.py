from box import Box
from trainer import Trainer
from tester import Tester

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

environment_file_name="Tennis.app"
#environment_file_name="Soccer.app"

def main(environment_file_name, hyperparameters, train_mode=False):
    if train_mode:
        trainer = Trainer(environment_file_name, hyperparameters)
        trainer.train()
        trainer.display_final_result()
    else:
        tester = Tester(environment_file_name, hyperparameters, games_to_play=10)
        tester.load_weights()
        tester.play()
    
# Launch the program:
train_mode = True
main(environment_file_name, hyperparameters, train_mode)