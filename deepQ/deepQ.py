# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Qingyang Feng (980940)
# Date:    05/22
# Purpose: Implemented the player agent which imports the trained deep Q policy for action selection

# IMPORTS ------------------------------------------------------------------------------------------------------------#

from template import Agent
import time
from Yinsh.yinsh_model import YinshGameRule
from   Yinsh.yinsh_utils import *
import torch
from agents.deepQ.deep_qfunction import DeepQFunction

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

THINKTIME = 0.95 # thinking time in seconds

# CLASS DEF ----------------------------------------------------------------------------------------------------------#

class myAgent(Agent):
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)
    
    def SelectAction(self, actions, state):
        start_time = time.time()
        deepQ = DeepQFunction(self.id, state_space=85, action_space=85+54*2+1, hiddem_dim=64)
        deepQ.q_network = torch.load("agents/deepQ/model100.pt")
        while time.time()-start_time < THINKTIME:
            (arg_max_q, _) = deepQ.get_max_q(state, actions, self.id)
            return arg_max_q
        #return random.choice(actions) # If no reward was found in the time limit, return a random action.
