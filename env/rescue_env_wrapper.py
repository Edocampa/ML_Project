import numpy as np
from pettingzoo.mpe import simple_tag_v3

class RescueEnvWrapper:
    def __init__(self, render_mode=None):
        self.env = simple_tag_v3.env(render_mode=render_mode)
        self.agents = None  # Placeholder
        self.possible_agents = None  # Placeholder
        
    def reset(self, seed=None, options=None):
        observations = self.env.reset(seed=seed, options=options)

        # Accedi alle informazioni dopo il reset
        self.agents = self.env.agents
        self.possible_agents = self.env.possible_agents

        # --- LOGICA CUSTOM PER VITTIME E TOOLS ---

        # Reset variabili custom
        self.victim_positions = []
        self.tool_positions = {}
        self.rescued_victims = set()
        
        # Posiziona N vittime in posizioni casuali
        num_victims = 1  # Puoi regolarlo
        self.victim_positions = self._sample_valid_positions(num_victims)

        # Posiziona uno o più tools (es. flashlight)
        num_tools = 1
        tool_coords = self._sample_valid_positions(num_tools)
        self.tool_positions["flashlight"] = tool_coords[0]  # Es. un solo flashlight

        return observations
    
    def _sample_valid_positions(self, n):
        """
        Restituisce n posizioni casuali valide nella griglia.
        In alternativa, puoi interrogare l'ambiente per ottenere la mappa reale.
        """
        grid_size = 5  # Adatta alla dimensione effettiva della mappa
        valid_positions = []

        while len(valid_positions) < n:
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            pos = (x, y)

            if pos not in valid_positions and not self._is_obstacle(pos):
                valid_positions.append(pos)

        return valid_positions
    
    def step(self, action):
        obs, reward, termination, truncation, info = self.env.last()
        done = termination or truncation

        if not done:
            self.env.step(action)
        else:
            return None

        # Post-step logic: check for victim rescue
        current_agent = self.env.agent_selection
        agent_pos = self.get_agent_position(current_agent)

        if self.is_agent_on_victim(agent_pos):
            reward += 10  # Reward for rescuing
            self.rescued_victims.add(agent_pos)

        return self.env.observe(current_agent), reward, termination, truncation, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def get_agent_position(self, agent):
        # Implement based on how your env stores agent positions
        return (0, 0)

    def is_agent_on_victim(self, pos):
        return pos in self.victim_positions
    def _is_obstacle(self, pos):
        # Placeholder — override this with real obstacle checking if needed
        return False  # Assume no obstacles for now

env = RescueEnvWrapper()
env.reset()

print("Vittime posizionate in:", env.victim_positions)
print("Tool 'flashlight' posizionato in:", env.tool_positions["flashlight"])