import pygame
import numpy as np
import os
import sys

# Constants
EMPTY = 0
WALL = 1
AGENT1 = 2
AGENT2 = 3
VICTIM = 4
ITEM = 5
FIRE = 6

COLORS = {
    EMPTY: (255, 255, 255),
    WALL: (50, 50, 50),
    FIRE: (255, 100, 0)
}

class SimpleGridWorld:
    def __init__(self, size=5, randomize=True):
        self.size = size
        self.cell_size = 60
        self.grid = np.zeros((size, size))

        # Track positions
        self.agent1_pos = None
        self.agent2_pos = None
        self.victim_pos = None
        self.item_pos = None
        self.wall_pos = None
        self.fire_pos = None

        # Track item possession
        self.agent1_has_item = False
        self.agent2_has_item = False

        # Initialize map
        if randomize:
            self._generate_random_map()
        else:
            self._setup_fixed_map()

        # Init Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("Multi-Agent Rescue Environment")
        self.clock = pygame.time.Clock()

        # Load assets
        self._load_assets()

    def _load_assets(self):
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")

        try:
            self.agent1_img = pygame.image.load(os.path.join(asset_dir, "robot.png"))
            self.agent2_img = pygame.image.load(os.path.join(asset_dir, "robot.png"))
            self.victim_img = pygame.image.load(os.path.join(asset_dir, "victim.png"))
            self.item_img = pygame.image.load(os.path.join(asset_dir, "item.png"))

            self.agent1_img = pygame.transform.scale(self.agent1_img, (self.cell_size, self.cell_size))
            self.agent2_img = pygame.transform.scale(self.agent2_img, (self.cell_size, self.cell_size))
            self.victim_img = pygame.transform.scale(self.victim_img, (self.cell_size, self.cell_size))
            self.item_img = pygame.transform.scale(self.item_img, (self.cell_size, self.cell_size))

        except Exception as e:
            print(f"Error loading images: {e}")
            sys.exit(1)

    def random_position(self, exclude=None):
        exclude = exclude or []
        while True:
            pos = (np.random.randint(self.size), np.random.randint(self.size))
            if pos not in exclude:
                return pos

    def _generate_random_map(self):
        occupied = []

        # Place objects
        self.agent1_pos = self.random_position(occupied)
        occupied.append(self.agent1_pos)

        self.agent2_pos = self.random_position(occupied)
        occupied.append(self.agent2_pos)

        self.victim_pos = self.random_position(occupied)
        occupied.append(self.victim_pos)

        self.item_pos = self.random_position(occupied)
        occupied.append(self.item_pos)

        self.wall_pos = self.random_position(occupied)
        occupied.append(self.wall_pos)

        self.fire_pos = self.random_position(occupied)

        # Reset grid
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.grid[self.agent1_pos] = AGENT1
        self.grid[self.agent2_pos] = AGENT2
        self.grid[self.victim_pos] = VICTIM
        self.grid[self.item_pos] = ITEM
        self.grid[self.wall_pos] = WALL
        self.grid[self.fire_pos] = FIRE

    def reset(self):
        """Reset environment with a new random map"""
        self.agent1_has_item = False
        self.agent2_has_item = False
        self._generate_random_map()
        return self.get_observations()

    def get_observations(self):
        return [self.agent1_pos, self.agent2_pos]

    def step(self, actions):
        a1, a2 = actions

        r1, d1 = self._move_agent(self.agent1_pos, a1, agent_id=1)
        r2, d2 = self._move_agent(self.agent2_pos, a2, agent_id=2)

        reward = r1 + r2
        done = d1 or d2

        if not done and self._can_rescue_victim():
            reward += 10
            done = True

        return self.get_observations(), [reward, reward], done, {}

    def _move_agent(self, pos, action, agent_id):
        x, y = pos
        if action == 0:   # Up
            new_x, new_y = max(0, x - 1), y
        elif action == 1: # Down
            new_x, new_y = min(self.size - 1, x + 1), y
        elif action == 2: # Left
            new_x, new_y = x, max(0, y - 1)
        elif action == 3: # Right
            new_x, new_y = x, min(self.size - 1, y + 1)

        new_pos = (new_x, new_y)

        # Check wall
        if self.grid[new_pos] == WALL:
            return   -1, False

        # Check item pickup
        if new_pos == self.item_pos:
            if agent_id == 1 and self.agent2_has_item == False:
                self.agent1_has_item = True
            elif agent_id == 2 and self.agent1_has_item == False:
                self.agent2_has_item = True
            self.grid[self.item_pos] = EMPTY  # Remove item

        # Update position
        if agent_id == 1:
            self.agent1_pos = new_pos
        else:
            self.agent2_pos = new_pos

        # Check fire
        if new_pos == self.fire_pos:
            return -10, True

        return 0, False

    def _can_rescue_victim(self):
        vx, vy = self.victim_pos
        x1, y1 = self.agent1_pos
        x2, y2 = self.agent2_pos

        d1 = abs(x1 - vx) + abs(y1 - vy)
        d2 = abs(x2 - vx) + abs(y2 - vy)

        near1 = d1 <= 1
        near2 = d2 <= 1

        if (near1 and near2) and (self.agent1_has_item or self.agent2_has_item):
            return True
        return False

    def render(self, delay=0.2):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        # Draw grid cells
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.grid[i, j]
                if cell_type == WALL:
                    pygame.draw.rect(self.screen, COLORS[WALL], rect)
                elif cell_type == FIRE:
                    pygame.draw.rect(self.screen, COLORS[FIRE], rect)
                else:
                    pygame.draw.rect(self.screen, COLORS[EMPTY], rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Draw victim
        vx, vy = self.victim_pos
        self.screen.blit(self.victim_img, (vy * self.cell_size, vx * self.cell_size))

        # Draw agents
        x1, y1 = self.agent1_pos
        self.screen.blit(self.agent1_img, (y1 * self.cell_size, x1 * self.cell_size))

        x2, y2 = self.agent2_pos
        self.screen.blit(self.agent2_img, (y2 * self.cell_size, x2 * self.cell_size))

        # Draw item if still present
        if self.grid[self.item_pos] == ITEM:
            ix, iy = self.item_pos
            self.screen.blit(self.item_img, (iy * self.cell_size, ix * self.cell_size))

        # Show text
        font = pygame.font.SysFont('Arial', 16)
        if self.agent1_has_item:
            self.screen.blit(font.render("Agent 1: Has Item", True, (0, 0, 0)), (5, 5))
        if self.agent2_has_item:
            self.screen.blit(font.render("Agent 2: Has Item", True, (0, 0, 0)), (5, 25))

        pygame.display.flip()
        pygame.time.wait(int(delay * 1000))
        
if __name__ == "__main__":
    env = SimpleGridWorld(size=5, randomize=True)
    obs = env.reset()
    done = False

    while not done:
        actions = [np.random.randint(4), np.random.randint(4)]
        next_obs, rewards, done, _ = env.step(actions)
        env.render(delay=0.3)
        print("Actions:", actions)
        print("Positions:", next_obs)
        print("Rewards:", rewards)

    pygame.quit()