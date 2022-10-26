<!-- ### Requirements

- **`python`** - `3.8`
- **`gym`** - `0.22`
- **`keras`** -  `2.4.3`
- **`tensorflow`** -  `2.2.0`

--- -->

# Project 1: Cart-Pole

<img src=cart-pole/cart_pole.gif width="400">

### Introduction

- The cartpole problem is an inverted pendelum problem where a stick is balanced upright on a cart. The cart can be moved left or right to and the goal is to keep the stick from falling over. A positive reward of +1 is received for every time step that the stick is upright. When it falls past a certain degree then the "episode" is over and a new one can begin.
- **`0`** - move cart to the left
- **`1`** - move cart to the right

- I solved this problem using DQN in around 60 episodes. Following is a graph of score vs episodes.

<img src=cart-pole/cart_pole.png width="400">

# Project 2: Mountain-Car

<img src=mountain-car/mountain_car.gif width="400">

### Introduction

- The Mountain Car problem is an environment where gravity exists (what a surprise) and the goal is to help a poor car win the battle against it.

The car needs to escape the valley where it got stuck. The car’s engine is not powerful enough to climb up the mountain in a single pass, so the only way to make it is to drive back and forth and build sufficient momentum.

Number of action spaces is 3. Action space is descrete in this environment.
- **`0`** - move car to left
- **`1`** - do nothing
- **`2`** - move car to right

- I solved this problem using DQN in around 15 episodes. Following is a graph of score vs episodes.

<img src=mountain-car/mountain_car.png width="400">


# Project 4: Bipedal-Walker

- BipedalWalker has 2 legs. Each leg has 2 joints. You have to teach the Bipedal-walker to walk by applying the torque on these joints. You can apply the torque in the range of (-1, 1). Positive reward is given for moving forward and small negative reward is given on applying torque on the motors.

### Smooth Terrain

- In the beginning, AI is behaving very randomly. It does not know how to control and balance the legs.

- After 600 episodes, it learns to maximize the rewards. It is walking in some different style. After all, it’s an AI not a Human. This is just one of the way to walk in order to get maximum reward. If I train it again, it might learn some other optimal way to walk.

<img src=bipedal-walker/training/1.gif width="400">

### Hardcore Terrain

- I saved my weight from the previous training on simple terrain and resumed my training on the hardcore terrain. I did it because the agent already knew how to walk on simple terrain and now it needs to learn how to cross obstacles while walking.

<img src=bipedal-walker/training/2.gif width="400">
