import gym
import random

# ROM https://github.com/openai/gym/releases/tag/v0.21.0

# env = gym.make('SpaceInvaders-v0', render_mode='human')
env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

#test environment setup works
episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0, 1, 2, 3, 4, 5])
        n_state, reward, done, info = env.step(action)
        score += reward
    print(f'Episode:{episode} Score:{score}')
env.close()