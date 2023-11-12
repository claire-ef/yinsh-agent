# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Qingyang Feng (980940)
# Date:    05/22
# Purpose: Implemented the agent used for training the neural network

# IMPORTS ------------------------------------------------------------------------------------------------------------#

from template import Agent
import random
import time
import copy
from statistics import mean
from Yinsh.yinsh_model import YinshGameRule
from   Yinsh.yinsh_utils import *

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

THINKTIME = 100 # thinking time in seconds

# CLASS DEF ----------------------------------------------------------------------------------------------------------#  

class myAgent(Agent):
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)
    
   # Carry out a given action on this state and return True if reward received.
    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score > score
    
    # get average length of sequences on board
    def avg_sequences_len(self, state):
        # possible lines on the board
        lines = []
        for pos in [(i, 5) for i in range(11)]:
            lines.append(self.game_rule.positionsOnLine(pos, 'd'))
            lines.append(self.game_rule.positionsOnLine(pos, 'h'))
        for pos in [(5, i) for i in range(11)]:
            lines.append(self.game_rule.positionsOnLine(pos, 'v'))
        # sequences
        seqs = set()
        agent_cntr = [CNTR_0, CNTR_1][self.id]
        board = state.board
        for line in lines:
            i = 0
            while i < len(line):
                j = i
                while j < len(line):
                    if board[line[j]] != agent_cntr:
                        if i != j and j-i >= 1:
                            seqs.add(frozenset(line[i:j]))
                        i = j+1
                        break
                    else:
                        j += 1
                i += 1
        if seqs:
            return mean([len(seq) for seq in seqs])
        return 0
    
    def get_reward(self, state, new_state):
        score = state.rings_won[self.id]
        new_score = new_state.rings_won[self.id]
        return new_score - score
    
    def get_potential(self, state):
        w0, w1, w2 = 0.01, 0.3, 0.01
        # Number of markers
        cntr_num = state.counters_left
        # average sequence length
        avg_seq_len = self.avg_sequences_len(state)
        # mobility: number of possible moves for this state
        state_copy = copy.deepcopy(state)
        action_num = len(self.game_rule.getLegalActions(state_copy, self.id))
        return w0*cntr_num + w1*avg_seq_len + w2*action_num

    def SelectAction(self, actions, state, qfunction):
        start_time = time.time()
        epsilon = 0.1
        while time.time()-start_time < THINKTIME:
            if random.random() < epsilon:
                return random.choice(actions)
            (arg_max_q, _) = qfunction.get_max_q(state, actions, self.id)
            return arg_max_q
        return random.choice(actions) # If no reward was found in the time limit, return a random action.
