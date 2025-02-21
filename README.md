# MLFB-Assignment1

## Introduction

  This project implements a maze exploration environment using Q-learning, a reinforcement learning algorithm. The maze is a 10x10 grid with traps and treasures. The agent learns to navigate from the top-left corner to the bottom-right corner while avoiding traps and collecting treasures. The environment is visualized using Pygame, and the learning progress is plotted using Matplotlib.

* Setup Environment: Visual Studio Code
## Requirements
* Python 3.x
* NumPy
* Pygame
* Matplotlib
## Code Structure
### 1. Maze class
  * Manages the maze environment, including traps, treasures, and the agent's state.
  * Methods:
    - reset(): Randomly places traps and treasures and resets the agent's position.
    - step(action): Updates the agent's state based on the chosen action and returns the new state, reward, and done flag.
    - render(score, action): Visualizes the maze using Pygame, showing the agent's position, traps, treasures, and the score.
### 2. QLearningAgent Class
* Implements the Q-learning algorithm.
* Methods:
  - choose_action(state): Selects an action based on the current state, using an epsilon-greedy policy.
  - learn(state, action, reward, next_state): Updates the Q-table based on the reward and next state.
  - decay_exploration(): Decays the exploration rate to gradually reduce randomness.
### 3. Helper Functions
* state_to_index(state): Converts a state tuple to an index.
* plot_convergence_curve(rewards): Plots the total reward per episode to visualize the learning progress.
### 4. Main Function
* Initializes the maze and agent.
* Runs the Q-learning loop, where the agent explores the maze, learns from rewards, and updates its Q-table.
* Renders the maze environment and handles Pygame events.
* Plots the convergence curve after training.
## Key Parameters
* WIDTH, HEIGHT: Dimensions of the maze grid (10x10).
* TRAP_PENALTY, TREASURE_REWARD, STEP_PENALTY: Reward values for traps, treasures, and steps.
* actions: List of possible actions (['up', 'down', 'left', 'right']).
* learning_rate, discount_factor, exploration_rate, exploration_decay: Q-learning hyperparameters.
## Video Link

  Welcome to my channel - https://www.youtube.com/@Bboji-v4j

  Display the video introduction of the MLFB-Assignment1 - [https://youtu.be/xgug7FXCUhg?si=LRx8oqZIrKEqz1RO ](https://youtu.be/5jyAIsQ6V70)

## Reference
  
