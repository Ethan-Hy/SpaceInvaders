# Space invaders Deep Reinforcment Learning project

### Description
Implementation of deep reinforcment learning to the Atari game Space Invaders. 

OpenAI gym (https://gym.openai.com/) is used for the environment for the Space Invaders game.

The deep learning model is built with Tensorflow Keras with Convolution2D and Dense layers. The layers funnel to save on computational resources.

The reinforment learning agent is built with Tensorflow Keras using EpsGreedyQPolicy combined with LinearAnnealedPolicy. Training was done initially for 90,000 steps and was continued to 500,000 steps. It is noted that this is far from the required steps needed to create a satisfactory solution, however, I am limitied by computational resources to reasonably train further.

### Requirements
The following are used: Python 3.7, Tensorflow 2.3.1, gym[atari], keras rl2.

### Instructions

`test.py` runs an initial test on the environment where the actions for the Space Invader game are randomly chosen.

`GPU_check.py` was a check that my GPU was being utilised by Tensorflow for processing.

`model.py` sets up the environment, builds the deep learning model and the reinforment learning agent. Here we train the model and it is set to save every 10,000 steps to allow for comparisons. The steps and interval values can be changed here to allow for further training. It can be currently seen that weights from 90,000 steps of training have been loaded in to continue training up to 500,000 steps.

`run.py` loads up the environemnt and model with specified weights, which are denoted by the folder they are saved into, which can be selected to the user's discretion. It is currently set to run 10 times and return a mean average score.

