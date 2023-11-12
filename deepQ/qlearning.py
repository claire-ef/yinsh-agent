# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Qingyang Feng (980940) adapted from the code for COMP90054 lecture notes
# Date:    05/22
# Purpose: Implemented the deep Q-learning Algorithm for Yinsh game

# IMPORTS ------------------------------------------------------------------------------------------------------------#

from Yinsh.yinsh_model import YinshGameRule as GameRule
from game import Game
import importlib
import copy
from tqdm import tqdm

# CLASS DEF ----------------------------------------------------------------------------------------------------------#  

class QLearning():
    def __init__(self, qfunction, alpha=0.1, discount_factor=0.9):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.qfunction = qfunction

    def execute(self, episodes=100):

        for _ in tqdm(range(episodes)):
            # load new game
            agent_0 = importlib.import_module('agents.t_073.trainAgent').myAgent(0)
            agent_1 = importlib.import_module('agents.t_073.alphaBeta').myAgent(1)
            gr = Game(GameRule,
                        [agent_0, agent_1],
                        num_of_agent = 2,
                        seed=12345,
                        time_limit=100,
                        warning_limit=100,
                        displayer=None,
                        interactive=False)

            agent_index = gr.game_rule.getCurrentAgentIndex()
            agent = gr.agents[agent_index]
            game_state = gr.game_rule.current_game_state
            game_state.agent_to_move = agent_index
            gs_copy = copy.deepcopy(game_state)

            actions = gr.game_rule.getLegalActions(game_state, agent_index)
            actions_copy = copy.deepcopy(actions)
            
            if agent_index == 0: # update if self agent
                selected = agent.SelectAction(actions_copy, gs_copy, self.qfunction)
                action = copy.deepcopy(selected)
            else:
                selected = agent.SelectAction(actions_copy, gs_copy)

            state = copy.deepcopy(game_state)

            # run game
            while not gr.game_rule.gameEnds():
                gr.game_rule.update(selected)

                agent_index = gr.game_rule.getCurrentAgentIndex()
                agent = gr.agents[agent_index]
                game_state = gr.game_rule.current_game_state
                game_state.agent_to_move = agent_index
                gs_copy = copy.deepcopy(game_state)

                actions = gr.game_rule.getLegalActions(game_state, agent_index)
                actions_copy = copy.deepcopy(actions)

                if agent_index == 0:
                    next_state = gr.game_rule.current_game_state
                    reward = agent.get_reward(state, next_state)
                    selected = agent.SelectAction(actions_copy, gs_copy, self.qfunction)
                    next_action = copy.deepcopy(selected)
                    q_value = self.qfunction.get_q_value(state, action, 0)
                    potential = agent.get_potential(state)
                    next_potential = agent.get_potential(next_state)
                    delta = self.get_delta(reward, q_value, potential, next_potential,
                                            state, next_state, next_action)
                    self.qfunction.update(state, action, delta)
                    state = next_state
                    action = next_action

                else:
                    selected = agent.SelectAction(actions_copy, gs_copy)
    
    """ Calculate the delta for the update """

    def get_delta(self, reward, q_value, potential, next_potential, state, next_state, next_action):
        next_state_value = self.state_value(next_state, next_action)
        delta = reward + self.discount_factor * (next_state_value + next_potential) - q_value - potential
        return self.alpha * delta

    """ Get the value of a state """
    def state_value(self, state, action):
        (_, max_q_value) = self.qfunction.get_max_q(state, [action], 0)
        return max_q_value


