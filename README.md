## Introduction

This is a deep reinforcement learning project which solves the Tennis unity environment using the DDPG algorithm.

Here is a preview of a well trained agent:
!["Trained Tennis Agent"](https://github.com/aivoric/RL-Tennis-Project/blob/main/images/tennis-example.gif?raw=true)

You can watch a fully trained agent play endlessly here:
https://youtu.be/7TBdqiHPXRI

Enjoy the read below for more details!

## Environment Introduction

Two agents control rackets to bounce a ball over a net. 

#### Reward Structure

Agent receives a reward of:
- +0.1 whenever it hits a ball over the net.
- -0.01 whenever it lets a ball hit the ground or hits the ball out of bounds

Hence the goal of each agent is to keep the ball in play.

#### Observation & Action Space

- 8 variables corresponding to the position and velocity of the ball and racket.
- Each agent receives its own, local observation.
- Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

#### Goal

- The task is episodic.
- The agents must get an average score of +0.5 over 100 consecutive episodes, after taking the maximum over both agents.

#### How Scoring Works

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

## How to Install the Project

To setup the environment on your machine:
1. Install Python 3.9+
2. Clone this repository:
        git clone https://github.com/aivoric/RL-Tennis-Project.git
3. Create a virtual python environment:
        python3 -m venv venv
4. Activate it with:
        source venv/bin/activate
5. Install all the dependecies:
        pip install -r requirements.txt
6. Download and install Anaconda so that you can run a Jupyter notebook from:
        http://anaconda.com/

## Overview of Project Files

The project is broken down into the following core python files:
- main.py: entry point into the program where you define hyperparameters and whether you are training/testing
- trainer.py: contains all the logic for training multiple agents within a selected environment
- tester.py: contains all the logic for testing trained agents within a selected environment
- environment.py: a wrapper for the unity environment with helper functions
- agent.py: DDPG agent logic which initialises deep neural networks, memory replay buffer, Ornstein-Uhlenbeck noise
- model.py: contains network architecture for the Actor and the Critic

The following folder:
- /results: contains subfolders for each iteration/experiment you run

And the following jupyter notebook which is used only for visualising the final scores across episodes:
- Tennis.ipynb

All the other files are used by the unity environment which allow you to run the environment. Most of the files are based on version 0.4 of ml-agents which is from July 2018 so it is quite outdated. For reference, a more modern ml-agents can be downloaded from: 
https://github.com/Unity-Technologies/ml-agents 

## How to Train a New Agent

1. Follow the instructions above for cloning the repo and installing depedencies.
2. Open main.py and configure your hyperparameters. Ensure **train_mode = True**
3. Start training with **python main.py** command in the console after activating the environment
4. Observe results on the console (see below for sample console results)
5. Wait for training to finish or interrupt training
6. Trained model weights will be saved in **result/results_i** where i is your experiment id defined in hyperparameters
7. Scores will be saved also in that folder into a **results.pickle** file
7. Visualise the scores by opening **Tennis.ipynb** and then adjusting the file location to the results.pickle.


## Example Console Output

!["Console Output Example"](https://github.com/aivoric/RL-Tennis-Project/blob/main/images/console-output-example.png?raw=true)