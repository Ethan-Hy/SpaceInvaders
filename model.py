import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(height, width, channels, actions)

# from https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
# Eps Greedy policy either:
#     - takes a random action with probability epsilon
#     - takes current best action with prob (1 - epsilon)
# Linear Annealing Policy computes a current threshold value and transfers it to an inner policy which chooses the action.
# The threshold value is following a linear function decreasing over time.


# eps from 1 to 0.1 over steps so agent initially explores and then uses what is knows more and more.
# value_test set to 0.05 (eps for testing) so agent still performs some random actions - ensures it cannot get stuck.
# build Keras-RL agent
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=500000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=90000)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))

callbacks = [ModelIntervalCheckpoint('weights/{step}/dqn_weights.h5f', interval=10000)]
callbacks += [FileLogger('weights/dqn_log.json', interval=100)]

dqn.load_weights('weights/90000/dqn_weights.h5f')
dqn.fit(env, callbacks=callbacks, nb_steps=500000, visualize=False, log_interval=100)
dqn.save_weights('weights/500000/dqn_weights.h5f')
