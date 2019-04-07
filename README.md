# pepperoni-reinforcement

## Abstract

When designing a bridge, it is desirable to minimize the material used while also meeting safety constraints. For the purpose of this work, a bridge is a two-dimensional rectangular shape with a polygonal hole cut into its body.

In this project, we propose two novel approaches:

1. To parameterize the shape of this hole via a circle packing. Ultimately, this is parameterized by the "radii of leading dancers", r_ld.

2. To use reinforcement learning to produce an agent to explore the space of possible r_ld, so as to find the optimal value.

Because a bridge-simulation environment is computationally expensive, we train the agent first on a much faster gradient-descent environment, and then allow it to fine-tune on the bridge.

As of yet, there are no results produced to put into this abstract.


## State of the Project

Recall: This is an optimization problem. The goal is to minimize the mass required to build a bridge, subject to a stress constraint. That is, we want to find the best *hole* shape for the bridge, parameterized by the $r_ld$, the radii of the leading dancers.

For future work, we hope we can produce an agent that can quickly learn how to find the best hole for different bridges.

See `example_rl.py` for the most pertinent code. In `gym_wrappers.py`, we utilize OpenAI's Gym framework to build an environment wrapper around the bridge and FEM/circle packing code. This provides an environment, `BHDEnv()`, which interfaces with agents and policies implemented in `keras-rl`. In this environment, the `action` provided is a small change vector which updates `rld` (the radii of the leading dancers).

The [DDPG algorithm](https://arxiv.org/pdf/1509.02971v2.pdf) is chosen as the continous analogue to DQNs.

### To be done:
 * Big blocker: Numerical issues in code resulting in NaN values.
 * Transfer learning: Train the agent first in a much-faster gradient-descent environment, then fine-tune it on the slow, bridge-design environment.
 * Optimize code, analyze using `cProfile`.
 * Enable in-training visualization plus results-generating code.


## Packages used:
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
