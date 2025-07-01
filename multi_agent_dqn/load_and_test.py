import torch
import numpy as np
from dqn_agent import QNetwork  # o da dove hai definito QNetwork

# 1) Iperparametri
state_dim = 6
n_actions = 4

# 2) Costruisci il modello e portalo in modalità eval
model = QNetwork(state_dim, n_actions)
model.load_state_dict(torch.load("dqn_weights.pth", map_location="cpu"))
model.eval()   # disabilita dropout, batch‐norm, ecc.

# 3) Prova una forward pass su uno stato di esempio
#    Ad esempio uno stato fittizio:
example_state = np.array([0,0,  1,2,  0,1], dtype=np.float32)
input_tensor  = torch.from_numpy(example_state).unsqueeze(0)  # forma [1,6]

with torch.no_grad():
    q_values = model(input_tensor)   # tensor [1,4]
    print("Q-values per le 4 azioni:", q_values.numpy().flatten())
    print("Azione consigliata:", int(q_values.argmax(dim=1).item()))
