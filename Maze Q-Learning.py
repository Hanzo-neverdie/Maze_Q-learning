import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
# Maze dimensions
WIDTH, HEIGHT = 10, 10  # 10x10 grid

# Define the rewards
TRAP_PENALTY = -20
TREASURE_REWARD = 60
STEP_PENALTY = -1

# Define the actions
actions = ['up', 'down', 'left', 'right']

# Initialize the Pygame module
pygame.init()

# Define the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set the display size
display_size = 400
cell_size = display_size // WIDTH
win = pygame.display.set_mode((display_size, display_size))
pygame.display.set_caption("Maze Q-learning")

# Define the maze class
class Maze:
    def __init__(self):
        self.maze = np.zeros((WIDTH, HEIGHT))
        self.state = (0, 0)  # Start at top-left corner
        self.terminal = (WIDTH-1, HEIGHT-1)  # End at bottom-right corner
        self.q_table = np.zeros((WIDTH, HEIGHT, len(actions)))

    def reset(self):
        # Randomly place traps and treasures
        self.traps = [(random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)) for _ in range(5) if (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)) not in [(0, 0), (WIDTH-1, HEIGHT-1)]]
        self.treasures = [(random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)) for _ in range(5) if (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)) not in [(0, 0), (WIDTH-1, HEIGHT-1)]]
        # Fixed traps and treasures
        #self.traps = [(1, 1), (3, 2), (5, 3), (7, 4), (9, 5)]
        #self.treasures = [(2, 2), (4, 3), (6, 4), (8, 5), (9, 6)]
        self.state = (0, 0)

    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < WIDTH-1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < HEIGHT-1:
            y += 1
        new_state = (x, y)
        self.state = new_state

        reward = STEP_PENALTY
        if new_state in self.traps:
            reward = TRAP_PENALTY
        elif new_state in self.treasures:
            reward = TREASURE_REWARD
            self.treasures.remove(new_state)  # Remove the treasure once it's obtained
        elif new_state == self.terminal:
            reward = 200  # High reward for reaching the terminal
            
        done = new_state == self.terminal
        return new_state, reward, done

    def render(self, score, action):
        win.fill(WHITE)
        for i in range(WIDTH):
            for j in range(HEIGHT):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(win, BLACK, rect, 1)
                if (i, j) == self.terminal:
                    pygame.draw.rect(win, BLUE, rect)
                elif (i, j) in self.traps:
                    pygame.draw.rect(win, RED, rect)
                elif (i, j) in self.treasures:
                    pygame.draw.rect(win, GREEN, rect)
        player_rect = pygame.Rect(self.state[1] * cell_size, self.state[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(win, (0, 0, 255), player_rect)

        # Display the score and action
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f'Score: {score}', True, BLACK)
        action_text = font.render(f'Action: {action}', True, BLACK)
        win.blit(score_text, (10, 10))
        win.blit(action_text, (10, 30))

        pygame.display.update()
    def state_to_index(self, state):
        return state[0], state[1]
        
class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.maze = maze
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(actions)
        else:
            state_index = self.maze.state_to_index(state)
            return actions[np.argmax(self.maze.q_table[state_index])]

    def learn(self, state, action, reward, next_state):
        state_index = self.maze.state_to_index(state)
        next_state_index = self.maze.state_to_index(next_state)
        action_index = actions.index(action)

        best_next_action = np.max(self.maze.q_table[next_state_index])
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.maze.q_table[state_index][action_index]

        self.maze.q_table[state_index][action_index] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

# Helper function to convert state to index
def state_to_index(state):
    return state[0], state[1]
def plot_convergence_curve(rewards):
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Convergence Curve")
    plt.legend()
    plt.show()
def main():
    maze = Maze()
    agent = QLearningAgent(maze)
    running = True
    score = 0
    episode_rewards = []

    while running:
        maze.reset()
        state = maze.state
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = maze.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Render the environment
            maze.render(total_reward, action)

            # Event handling
            for event in pygame.event.get():    
                if event.type == pygame.QUIT:
                    running = False

            # Delay to control the game speed
            pygame.time.delay(100)
        episode_rewards.append(total_reward)
        agent.decay_exploration()
        score += total_reward
        if len(episode_rewards) >= 60:  
            running = False
    plot_convergence_curve(episode_rewards)
if __name__ == "__main__":
    main()