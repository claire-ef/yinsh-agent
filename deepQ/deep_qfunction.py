# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Qingyang Feng (980940) adapted from the code for COMP90054 lecture notes
# Date:    05/22
# Purpose: Implemented deep Q-learning function for Yinsh game

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from   Yinsh.yinsh_utils import *

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

RING_SELF   = 1
CNTR_SELF   = 2
RING_OPPO   = -1
CNTR_OPPO   = -2
LEGAL_POS = [pos for pos in [(i, j) for i in range(11) for j in range(11)] if pos not in ILLEGAL_POS]

# CLASS DEF ----------------------------------------------------------------------------------------------------------#       

class DeepQFunction():

    def __init__(
        self, agent_id, state_space, action_space, hiddem_dim=64, alpha=0.01
    ) -> None:
        self.agent_id = agent_id
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha

        # Create a sequential neural network to represent the Q function
        self.q_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hiddem_dim),
            nn.ReLU(),
            nn.Linear(in_features=hiddem_dim, out_features=hiddem_dim),
            nn.ReLU(),
            nn.Linear(in_features=hiddem_dim, out_features=self.action_space),
        )
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        
    def update(self, state, action, delta):
        # Train the network based on the squared error.
        self.optimiser.zero_grad()  # Reset gradients to zero
        (delta ** 2).backward()  # Back-propagate the loss through the network
        self.optimiser.step()  # Do a gradient descent step with the optimiser

    def get_q_value(self, state, action, agent_id, for_max=False):
        # Convert the state into a tensor
        state = self.encode_state(state, agent_id)
        q_values = self.q_network(state)
        q_value = q_values[self.encode_action(action)]  # Index q-values by action
        return q_value

    """ Return a pair containing the action and Q-value, where the
        action has the maximum Q-value in state
    """

    def get_max_q(self, state, actions, agent_id):
        # Convert the state into a tensor
        state = torch.as_tensor(self.encode_state(state, agent_id), dtype=torch.float32)
        # Since we have a multi-headed q-function, we only need to pass through the network once
        # call torch.no_grad() to avoid tracking the gradients for this network forward pass
        with torch.no_grad():
            q_values = self.q_network(state)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = q_values[self.encode_action(action)].item()  
            if max_q < value:
                arg_max_q = action
                max_q = value
        return (arg_max_q, max_q)
    
    """
    Encode the state according to the agent_id and Turn the state into a tensor.
    """
    def encode_state(self, state, agent_id):
        flattened = []
        for pos in LEGAL_POS:
            flattened.append(state.board[pos])
        if agent_id == 0:
            self_ring = RING_0
            self_cntr = CNTR_0
            oppo_ring = RING_1
            oppo_cntr = CNTR_1
        else:
            self_ring = RING_1
            self_cntr = CNTR_1
            oppo_ring = RING_0
            oppo_cntr = CNTR_0
        # change values to oppo/self ring/cntr
        flattened = np.where(flattened != oppo_ring, flattened, RING_OPPO)
        flattened = np.where(flattened != oppo_cntr, flattened, CNTR_OPPO)
        flattened = np.where(flattened != self_ring, flattened, RING_SELF)
        flattened = np.where(flattened != self_cntr, flattened, CNTR_SELF)
        return torch.as_tensor(flattened, dtype=torch.float32)
    
    """
    Encode the action: given action, return the encoded action id
        0-84: 'place ring'
        85+0 - 85+53: 'place and move'
        85+54+0 - 85+54+53: 'place, move, remove'
        85+54+54: 'pass'
    """
    def encode_action(self, action):
        if action["type"] == "place ring":
            return LEGAL_POS.index(action['place pos']) # 85 actions

        elif action["type"] == "place and move":
            ppos, mpos = action['place pos'], action['move pos']
            line_id = None # 3 different values
            if ppos[0] == mpos[0]:
                line_id = 0 # horizontal
                step = mpos[1]-ppos[1]
            elif ppos[1] == mpos[1]:
                line_id = 1 # vertical
                step = mpos[0]-ppos[0]
            else:
                line_id = 2 # diagonal
                step = mpos[0]-ppos[0]
            step = (step + 9) if step < 0 else (step + 8) # 18 different values
            return 85 + line_id*18 + step

        elif action["type"] == "place, move, remove":
            ppos, mpos = action['place pos'], action['move pos']
            line_id = None # 3 different values
            if ppos[0] == mpos[0]:
                line_id = 0 # horizontal
                step = mpos[1]-ppos[1]
            elif ppos[1] == mpos[1]:
                line_id = 1 # vertical
                step = mpos[0]-ppos[0]
            else:
                line_id = 2 # diagonal
                step = mpos[0]-ppos[0]
            step = (step + 9) if step < 0 else (step + 8) # 18 different values
            return 85 + 54 + line_id*18 + step
        else:
            return 85 + 54 + 54
        
    """ Extract a policy for this Q-function  """
    def extract_policy(self):
        return self.q_network
    