import numpy as np
import gym
from gym_wrappers import BHDEnv

# See: https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


env = BHDEnv(length = 20.0, height = 10.0, allowable_stress = 200.0)
ob = env.reset()
np.random.seed(123)
env.seed(123)
ld_length = env.action_space.shape[0]

model = Sequential()
# TODO - Hope the model can Flatten input spaces.Dict,
#        and learn the input shape automatically.
#        Else, we'll need to convert the dict to a numpy array / shapes.Box
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(ld_length))

memory = SequentialMemory(limit=50, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=ld_length, memory=memory, nb_steps_warmup=3,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
