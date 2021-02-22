## Summary

### Results

The environment was solved after 758 episodes using the DDPG algorithm for training the agents.

A top score of 0.96 (100 episode average) at about 1000 episodes.

The training was done on CPU on a Macbook Pro 2.6 GHz 6-Core Intel Core i7. It took about 2 hours.

A graphic summarising the performance:
!["Tennis Scores"](https://github.com/aivoric/RL-Tennis-Project/blob/main/images/tennis_scores.png?raw=true)

The solution model weights can be found in the /results/results_7 folder. File names start with **solved_**

The best trained agent can also be found in the same /results/results_7 folder but without the **solved_** at the beginning of file names.

A video of the best trained agent can be found below. The best trained agents have learned to play an endless game of tennis!
https://youtu.be/7TBdqiHPXRI

The console output of the training:

!["Console Output"](https://github.com/aivoric/RL-Tennis-Project/blob/main/images/console-output-example.png?raw=true)

### Learning Algorithm

The agents were trained using a DDPG algorithm which utilises a Replay buffer.

The learning process follows:
* Initialise 2 identical DDPG agents (one for each racket)
* Initialise a Replay Buffer
* Initialise a 
* Initialise an environment and get the initial state
* Begin episode loop for N episodes:
    - Reset agents and environment
    - Begin episode stepping loop:
        - Pass the state to the agents and get their action
        - Use the action in the environment and step through it
        - Collect next state, rewards, and dones (if episode is finished)
        - Give the agents the obtained states and rewards information
        - Agents save information in a replay buffer
        - Agents train by sampling experiences from the replay buffer
        - Update scores
    - Update episode scores
    - Save models if required


### Model Architecture

2 slighly different network architectures are used for the actor and the agent.

#### Actor
- State size -> 512 hidden cells with ReLU activation function
- 512 cells -> 256 cells with ReLU activation function
- 256 cells -> 1 action with Tanh activation function

#### Critic
- State size -> 512 hidden cells with ReLU activation function
- 20% dropout probability in the hidden layer of 512 cells
- 512 cells + Action -> 1 value with ReLU activation function

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