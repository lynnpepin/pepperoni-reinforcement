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


# Input shape
# TODO - This will still be size (34,), or maybe (1,)? But can no longer be a dictionary...
obs = env.observation_space
space_shapes = [obs.spaces[key].shape for key in obs.spaces.keys()]
#e.g. spaces_shapes = [(10,), (1,), (1,), (10, 2), (1,), (1,)]
input_shape = (np.sum([np.prod(shape) for shape in space_shapes]), )

# Output shape
ld_length = env.action_space.shape[0]


# Set up neural network, RL stuff
model = Sequential()
model.add(Dense(32, input_shape = input_shape))
model.add(Activation('relu'))
model.add(Dense(ld_length))

memory = SequentialMemory(limit=50, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=ld_length, memory=memory, nb_steps_warmup=3,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# And fit!
dqn.fit(env, nb_steps=3, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
