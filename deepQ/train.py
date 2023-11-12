 #INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Qingyang Feng (980940)
# Date:    05/22
# Purpose: Script written for training neural network

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import torch
from deep_qfunction import DeepQFunction
from qlearning import QLearning

# Code ---------------------------------------------------------------------------------------------------------------#

qfunction = DeepQFunction(0, state_space=85, action_space=85+54*2+1, hiddem_dim=64)
QLearning(qfunction, alpha=0.1, discount_factor=0.9).execute(episodes=1000)
policy = qfunction.extract_policy()
torch.save(policy, "agents/group73/policy.pt")