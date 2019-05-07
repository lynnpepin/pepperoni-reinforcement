"""example_rl.py

A really simple example / `test run' of learning with our BHDEnv.
Uses the "continous DQN" DDPG agent.
Ideally, this should run without errors. But right now, it doesn't. See README.

Code shamelessly adapted from example: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py
"""

import numpy as np
import gym
import pickle
from gym_wrappers import BHDEnv
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, LeakyReLU
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import TrainIntervalLogger

# Set up environment and input/output shapes.
env = BHDEnv(length=20.0, height=10.0, allowable_stress=200.0)
ob = env.reset()
np.random.seed(123)
env.seed(123)

obs = env.observation_space
ld_length = env.action_space.shape[0]
nb_actions = ld_length
input_shape = (1,) + obs.shape

# Set up neural networks required for DDPG, i.e. continuous-space DQN
actor = Sequential()
actor.add(Flatten(input_shape=input_shape))
actor.add(Dense(64))
actor.add(LeakyReLU(alpha=0.0625))
actor.add(Dense(64))
actor.add(LeakyReLU(alpha=0.0625))
actor.add(Dense(nb_actions, activation="tanh"))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=input_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(1)(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Compile and fit agent:
memory = SequentialMemory(limit=10, window_length=1)

# OrnsteinUhlen process is a type of random walk. Explore the space R^size
# Each step:
# x += x_prev + theta*(mu-x_prev)*dt + current_sigma*sqrt(dt)*Normal(in R^size)
#   theta: strength of attraction to return to cnter
random_process = OrnsteinUhlenbeckProcess(
    size=nb_actions, theta=.15, mu=0., sigma=.3)

# DDPGAgent:
#   Only few warm-up steps before starting to train actor and critic.
#       Agent should be able to immediately start moving towards an ideal solution!
#   target_model_update = .01. Smooth averaging coefficient.
#       See: https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html
#   
agent = DDPGAgent(
    nb_actions=nb_actions,
    actor=actor,
    nb_steps_warmup_actor=100,
    critic=critic,
    critic_action_input=action_input,
    nb_steps_warmup_critic=100,
    memory=memory,
    random_process=random_process,
    gamma=.99,
    target_model_update=.01)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# nb_max_episode_steps = max steps per episode before resetting
# nb_steps = number of training steps to be performed.
# No idea what the difference between the two are tbh!
# See: https://github.com/keras-rl/keras-rl/blob/master/rl/core.py
train_callback = TrainIntervalLogger()

agent.load_weights("ddpg_example-rl_weights")

history = agent.fit(env, nb_steps=10000, visualize=False, verbose=1, nb_max_episode_steps=400,
                    callbacks = [train_callback,])

# Saving infos
with open('train_callback.infos.pickle', 'wb') as handle:
    pickle.dump(train_callback.infos, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('train_callback.episode_rewards.pickle', 'wb') as handle:
    pickle.dump(train_callback.episode_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
# Example loading:
import numpy as np
import gym
import pickle

with open('train_callback.infos.pickle', 'rb') as handle:
    train_callbacks_infos = pickle.load(handle)

with open('train_callback.episode_rewards.pickle', 'rb') as handle:
    train_callbacks_episode_rewards = pickle.load(handle)

infos = np.array(train_callbacks_infos)
rewards = np.array(train_callbacks_episode_rewards)
'''

# Post-training
agent.save_weights('ddpg_example-rl_weights.h5f', overwrite=True)
agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=400)


