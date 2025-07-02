import pygame
import numpy as np
import os
import sys

EMPTY = 0
WALL = 1
AGENT = 2
VICTIM = 3
ITEM = 4
FIRE = 5

COLORS = {
    EMPTY: (255, 255, 255),
    WALL: (50, 50, 50),
    FIRE: (255, 100, 0)
}

class SimpleSingleAgentEnv:
    def __init__(self, size=5, randomize=True):
        self.size = size
        self.cell_size = 60
        self.grid = np.zeros((size, size))
        self.agent_pos = None
        self.victim_pos = None
        self.item_pos = None
        self.wall_pos = None
        self.fire_pos = None
        self.agent_has_item = False

        if randomize:
            self._generate_random_map()
        else:
            self._setup_fixed_map()

        pygame.init()
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("Single-Agent Rescue Environment")
        self.clock = pygame.time.Clock()
        self._load_assets()

    def _load_assets(self):
        asset_dir = os.path.join(os.path.dirname(__file__), "../env/assets")

        try:
            self.agent_img = pygame.image.load(os.path.join(asset_dir, "robot.png"))
            self.victim_img = pygame.image.load(os.path.join(asset_dir, "victim.png"))
            self.item_img = pygame.image.load(os.path.join(asset_dir, "item.png"))
            self.fire_img = pygame.image.load(os.path.join(asset_dir, "fire.png"))
            self.wall_img = pygame.image.load(os.path.join(asset_dir, "wall.png"))

            for attr in ['agent_img', 'victim_img', 'item_img', 'fire_img', 'wall_img']:
                setattr(self, attr, pygame.transform.scale(getattr(self, attr), (self.cell_size, self.cell_size)))

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
        self.victim_pos = (3, 1)
        occupied.append(self.victim_pos)

        self.wall_pos = (1, 3)
        occupied.append(self.wall_pos)

        self.fire_pos = (4, 2)
        occupied.append(self.fire_pos)

        self.agent_pos = self.random_position(occupied)
        occupied.append(self.agent_pos)

        self.item_pos = self.random_position(occupied)
        occupied.append(self.item_pos)

        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.grid[self.agent_pos] = AGENT
        self.grid[self.victim_pos] = VICTIM
        self.grid[self.item_pos] = ITEM
        self.grid[self.wall_pos] = WALL
        self.grid[self.fire_pos] = FIRE

    def reset(self):
        self.agent_has_item = False
        self._generate_random_map()
        return self.get_observation()

    def get_observation(self):
        return self.agent_pos

    def step(self, action):
        reward, done = self._move_agent(action)

        if done:
            return self.get_observation(), reward, True, {}

        if self._can_rescue_victim():
            return self.get_observation(), 10, True, {}

        return self.get_observation(), reward, False, {}

    def _move_agent(self, action):
        x, y = self.agent_pos
        if action == 0:  # Up
            new_x, new_y = max(0, x - 1), y
        elif action == 1:  # Down
            new_x, new_y = min(self.size - 1, x + 1), y
        elif action == 2:  # Left
            new_x, new_y = x, max(0, y - 1)
        elif action == 3:  # Right
            new_x, new_y = x, min(self.size - 1, y + 1)

        new_pos = (new_x, new_y)

        if self.grid[new_pos] == WALL:
            return -1, False

        if new_pos == self.item_pos:
            self.agent_has_item = True
            self.grid[self.item_pos] = EMPTY

        self.agent_pos = new_pos

        if new_pos == self.fire_pos:
            return -10, True

        return 0, False

    def _can_rescue_victim(self):
        ax, ay = self.agent_pos
        vx, vy = self.victim_pos
        dist = abs(ax - vx) + abs(ay - vy)
        return self.agent_has_item and dist <= 1

    def render(self, delay=0.2):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.grid[i, j]
                if cell_type == WALL:
                    self.screen.blit(self.wall_img, (j * self.cell_size, i * self.cell_size))
                elif cell_type == FIRE:
                    self.screen.blit(self.fire_img, (j * self.cell_size, i * self.cell_size))
                else:
                    pygame.draw.rect(self.screen, COLORS[EMPTY], rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Draw static items
        vx, vy = self.victim_pos
        self.screen.blit(self.victim_img, (vy * self.cell_size, vx * self.cell_size))

        if self.grid[self.item_pos] == ITEM:
            ix, iy = self.item_pos
            self.screen.blit(self.item_img, (iy * self.cell_size, ix * self.cell_size))

        # Draw agent
        ax, ay = self.agent_pos
        self.screen.blit(self.agent_img, (ay * self.cell_size, ax * self.cell_size))

        # HUD
        font = pygame.font.SysFont('Arial', 16)
        if self.agent_has_item:
            self.screen.blit(font.render("Agent: Has Item", True, (0, 0, 0)), (5, 5))

        pygame.display.flip()
        pygame.time.wait(int(delay * 1000))


if __name__ == "__main__":
    env = SimpleSingleAgentEnv(size=5)
    obs = env.reset()
    done = False

    while not done:
        action = np.random.randint(4)
        next_obs, reward, done, _ = env.step(action)
        env.render(0.3)
        print("Action:", action, "Next:", next_obs, "Reward:", reward)

    pygame.quit()
