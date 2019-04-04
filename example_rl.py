# See: https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py

import numpy as np
import gym
from gym_wrappers import BHDEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# Set up environment
env = BHDEnv(length = 20.0, height = 10.0, allowable_stress = 200.0)
ob = env.reset()
np.random.seed(123)
env.seed(123)


# Input shape and output shape
obs = env.observation_space
ld_length = env.action_space.shape[0]
input_shape = (1,) + obs.shape
#input_shape = (1,) * (ld_length - 1) + obs.shape


# Set up neural network, RL stuff
model = Sequential()
model.add(Dense(32, input_shape = input_shape))
model.add(Activation('relu'))
model.add(Dense(ld_length))
model.add(Flatten())


memory = SequentialMemory(limit=50, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=ld_length, memory=memory, nb_steps_warmup=3,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# And fit!
dqn.fit(env, nb_steps=2, visualize=False, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
dqn.test(env, nb_episodes=5, visualize=True)
