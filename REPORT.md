## Summary

### Results

The environment was solved after 758 episodes. A top score of 0.96 (100 episode average) at about 1000 episodes.

The training was done on CPU on a Macbook Pro 2.6 GHz 6-Core Intel Core i7. It took about 2 hours.

A graphic summarising the performance:
!["Tennis Scores"](https://github.com/aivoric/RL-Tennis-Project/blob/main/images/tennis_scores.png?raw=true)

The solution model weights can be found in the /results/results_7 folder. File names start with **solved_**

The best trained agent can also be found in the same folder /results/results_7 folder but without the **solved_** at the beginning of file names.

A video of the best trained agent can be found below. The best trained agents have learned to play an endless game of tennis!
https://youtu.be/7TBdqiHPXRI

The console output of the training:

!["Console Output"](https://github.com/aivoric/RL-Tennis-Project/blob/main/images/console-output-example.png?raw=true)

### Learning Algorithm

Coming soon!


### Model Architecture

Coming soon!

### Hyperparameters

    'ITERATION': 7,               # Iteration ID allows to put results into new folders
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

## Future Improvements

- Experiment with Prioritised Experience Replay. It has been shown to "can reduce the training time and improve the stability of the training process, and is less sensitive to the changes of some hyperparameters such as the size of replay buffer, minibatch and the updating rate of the target network". https://www.semanticscholar.org/paper/A-novel-DDPG-method-with-prioritized-experience-Hou-Liu/027d002d205e49989d734603ff0c2f7cbfa6b6dd
- Experiment with Sigma decay in the Ornstein-Uhlenbeck process. At first, training was attempted with sigma decay but it didn't work very well, so a static sigma value was used. I think more effort needs to be taken in order to identify how sigma
- Improve the networks with Batch normalisation as this can speed up and stabilise training. 
- Experiment with different network architectures. Initially an architecture of 256 input cells and 128 hidden cells was used. However, this showed poor performance, so it was expanded to 512 input cells and 256 hidden cells. This worked better. However, further experimentation is required.
- Experiment with different values for TAU. TAU was initially really low (0.005) and training was extremely slow. Once TAU was increased to 0.15 the training became much faster.