# See: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_pendulum.py

import numpy as np
import gym
from gym_wrappers import BHDEnv
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# Set up environment and input/output shapes.
env = BHDEnv(length = 20.0, height = 10.0, allowable_stress = 200.0)
ob = env.reset()
np.random.seed(123)
env.seed(123)

obs = env.observation_space
ld_length = env.action_space.shape[0]
nb_actions = ld_length
input_shape = (1,) + obs.shape


# Set up neural network for DDPG
actor = Sequential()
actor.add(Flatten(input_shape=input_shape))
actor.add(Dense(16, activation = 'relu'))
actor.add(Dense(16, activation = 'relu'))
actor.add(Dense(16, activation = 'relu'))
actor.add(Dense(nb_actions))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=input_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
x = Dense(1)(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Compile and fit agent.
memory = SequentialMemory(limit=100, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, nb_steps_warmup_actor=100,
                  critic=critic, critic_action_input=action_input, nb_steps_warmup_critic=100,
                  memory=memory, random_process=random_process,
                  gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=3, visualize=False, verbose=1, nb_max_episode_steps=2)

# Post-training
agent.save_weights('ddpg_example-rl_weights.h5f', overwrite=True)
agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=2)
