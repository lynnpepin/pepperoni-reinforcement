# pepperoni-reinforcement

## Uses:
 * Built-in libraries: math, random, collections, unittest,
 * numpy (1.16.2)
 * sklearn (0.20.2)
 * matplotlib (3.0.0)
 * tensorflow (1.12.0)
 * keras (2.2.4)
 * h5py (2.9.0)
 * gym (OpenAI gym) (0.12.1)
 * rl (keras-rl) (https://github.com/keras-rl/keras-rl)

## Files:
 * **reinforcement_utils.py** : A mostly deprecated wasteland of old code that should be cleaned up, plus some preprocessing functions. Most of this functionality is provided by gym_wrappers now.
 * **gym_wrappers.py** : Provides OpenAI Gym.Env wrapper for BridgeHoleDesign, plus necessary preprocessing functions.
 * **pepperoni.py** : Provides BridgeHoleDesign() environment.
 * **_FEM.py** : Provides finite element method analysis for pepperoni.

## Run-down:
See our original write up for more details.

We want to minimize the **mass** of a bridge subject to a constant downward force, subject to a **stress** constraint.

The **reward** is maximum\_mass - mass. The **environment** is the bridge (see: gym\_wrapper.BHDEnv()).

## How to use:
We don't have an actor in master yet. Check out rl_example.py in the rl-example branch if you're interested in a minimum working example.

Also, see gym_wrappers.BHDEnv(), and check out Open AI's Gym. BHDEnv() is the Environment the agent will be interacting with.
