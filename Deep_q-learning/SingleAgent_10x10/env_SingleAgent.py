import pygame
import numpy as np
import os
import sys

# Tile types
EMPTY = 0
WALL = 1
AGENT = 2
VICTIM = 3
ITEM = 4
FIRE = 5

# Colors for rendering simple cells
COLORS = {
    EMPTY: (255, 255, 255),
    WALL: (50, 50, 50),
    FIRE: (255, 100, 0)
}

class SimpleSingleAgentEnv:


    _STOCHASTIC_MOVES = {
        0: [(0.90, -1, 0),  (0.05, -1, +1), (0.05, -1, -1)],  # Up
        1: [(0.90, +1, 0),  (0.05, +1, +1), (0.05, +1, -1)],  # Down
        2: [(0.90,  0, -1), (0.05, -1, -1), (0.05, +1, -1)],  # Left
        3: [(0.90,  0, +1), (0.05, -1, +1), (0.05, +1, +1)]   # Right
    }


    def __init__(self, size=10, randomize=False):
        self.size = size
        self.cell_size = 60
        self.grid = np.zeros((size, size), dtype=int)
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
        self.screen = pygame.display.set_mode((self.size * self.cell_size,
                                               self.size * self.cell_size))
        pygame.display.set_caption("Single-Agent Rescue Environment")
        self.clock = pygame.time.Clock()
        self._load_assets()

    def _load_assets(self):
        asset_dir = os.path.join(os.path.dirname(__file__), "../../env/assets")
        try:
            self.agent_img   = pygame.image.load(os.path.join(asset_dir, "robot.png"))
            self.victim_img  = pygame.image.load(os.path.join(asset_dir, "victim.png"))
            self.item_img    = pygame.image.load(os.path.join(asset_dir, "item.png"))
            self.fire_img    = pygame.image.load(os.path.join(asset_dir, "fire.png"))
            self.wall_img    = pygame.image.load(os.path.join(asset_dir, "wall.png"))
            for attr in ['agent_img', 'victim_img', 'item_img', 'fire_img', 'wall_img']:
                img = getattr(self, attr)
                setattr(self, attr, pygame.transform.scale(img, (self.cell_size, self.cell_size)))
        except Exception as e:
            print(f"Error loading images: {e}")
            sys.exit(1)

    def random_position(self, exclude=None):
        exclude = exclude or []
        while True:
            pos = (np.random.randint(self.size), np.random.randint(self.size))
            if pos not in exclude:
                return pos

    def _setup_fixed_map(self):
        self.victim_pos = (8, 1)
        self.wall_pos   = (7, 1)
        self.fire_pos   = (9, 2)
        self.agent_pos  = (0, 0)
        self.item_pos   = (5, 4)

        self.grid.fill(EMPTY)
        self.grid[self.agent_pos] = AGENT
        self.grid[self.victim_pos] = VICTIM
        self.grid[self.item_pos]   = ITEM
        self.grid[self.wall_pos]   = WALL
        self.grid[self.fire_pos]   = FIRE

    def _generate_random_map(self):
        occupied = []
        # start with fixed hazards
        self.victim_pos = (3, 1); occupied.append(self.victim_pos)
        self.wall_pos   = (1, 3); occupied.append(self.wall_pos)
        self.fire_pos   = (4, 2); occupied.append(self.fire_pos)

        # randomize agent and item
        self.agent_pos  = self.random_position(occupied); occupied.append(self.agent_pos)
        self.item_pos   = self.random_position(occupied); occupied.append(self.item_pos)

        self.grid.fill(EMPTY)
        self.grid[self.agent_pos] = AGENT
        self.grid[self.victim_pos] = VICTIM
        self.grid[self.item_pos]   = ITEM
        self.grid[self.wall_pos]   = WALL
        self.grid[self.fire_pos]   = FIRE

    def reset(self):
        self.agent_has_item = False
        # rebuild fixed map state
        self._setup_fixed_map()
        return self.get_observation()

    def get_observation(self):
        return self.agent_pos

    def step(self, action):
        reward, done = self._move_agent(action)
        if done:
            return self.get_observation(), reward, True, {}
        if self.agent_has_item and abs(self.agent_pos[0] - self.victim_pos[0]) + \
           abs(self.agent_pos[1] - self.victim_pos[1]) <= 1:
            return self.get_observation(), 10, True, {}
        return self.get_observation(), reward, False, {}

    def _move_agent(self, action):
        # 1) scegli la deviazione casuale in base alle probabilità
        r = np.random.rand()
        cum = 0.0
        for p, dx, dy in self._STOCHASTIC_MOVES[action]:
            cum += p
            if r <= cum:
                break

        # 2) calcola la nuova posizione proposta (clippata ai bordi)
        x, y = self.agent_pos
        new_x = np.clip(x + dx, 0, self.size - 1)
        new_y = np.clip(y + dy, 0, self.size - 1)
        new_pos = (new_x, new_y)

        # 3) logica di interazione identica a prima
        if self.grid[new_pos] == WALL:
            return -1, False

        if new_pos == self.item_pos:
            self.agent_has_item = True
            self.grid[self.item_pos] = EMPTY

        self.agent_pos = new_pos

        if new_pos == self.fire_pos:
            return -10, True

        return -0.1, False

    def render(self, delay=0.2):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        self.screen.fill((255,255,255))
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size,
                                   self.cell_size, self.cell_size)
                if self.grid[i,j] == WALL:
                    self.screen.blit(self.wall_img, rect)
                elif self.grid[i,j] == FIRE:
                    self.screen.blit(self.fire_img, rect)
                else:
                    pygame.draw.rect(self.screen, COLORS[EMPTY], rect)
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)
        # static victim
        vx, vy = self.victim_pos
        self.screen.blit(self.victim_img, (vy*self.cell_size, vx*self.cell_size))
        # item (if not picked)
        if self.grid[self.item_pos] == ITEM:
            ix, iy = self.item_pos
            self.screen.blit(self.item_img, (iy*self.cell_size, ix*self.cell_size))
        # agent
        ax, ay = self.agent_pos
        self.screen.blit(self.agent_img, (ay*self.cell_size, ax*self.cell_size))
        # HUD
        font = pygame.font.SysFont('Arial', 16)
        if self.agent_has_item:
            self.screen.blit(font.render("Agent: Has Item", True, (0,0,0)), (5,5))
        pygame.display.flip()
        pygame.time.wait(int(delay*1000))

if __name__ == "__main__":
    env = SimpleSingleAgentEnv(size=10, randomize=False)
    done = False
    obs = env.reset()
    while not done:
        action = np.random.randint(4)
        obs, reward, done, _ = env.step(action)
        env.render(delay=0.3)
    pygame.quit()
