# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import math
import util
from game import Agent
from game import Directions
from searchAgents import mazeDistance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.value(gameState, 0, 0)[1]

    def value(self, game_state, depth, agent_index):
        if not game_state.getLegalActions(
                agent_index) or depth == self.depth or game_state.isWin() or game_state.isLose():
            return [self.evaluationFunction(game_state), 0]

        # If the agent is the last ghost, increment the depth and set the next agent to pacman
        if agent_index == game_state.getNumAgents() - 1:
            depth += 1
            next_agent_index = 0
        # Otherwise, set the next agent to the next ghost
        else:
            next_agent_index = agent_index + 1

        if agent_index == 0:
            return self.max_value(game_state, depth, agent_index, next_agent_index)
        else:
            return self.min_value(game_state, depth, agent_index, next_agent_index)

    def max_value(self, game_state, depth, agent_index, next_agent_index):
        v = -math.inf
        best_action = Directions.STOP
        for action in game_state.getLegalActions(agent_index):
            previous_v = v
            next_game_state = game_state.generateSuccessor(agent_index, action)
            new_value = self.value(next_game_state, depth, next_agent_index)[0]
            v = max(v, new_value)
            if previous_v != v:
                best_action = action
        return [v, best_action]

    def min_value(self, game_state, depth, agent_index, next_agent_index):
        v = math.inf
        best_action = Directions.STOP
        for action in game_state.getLegalActions(agent_index):
            previous_v = v
            next_game_state = game_state.generateSuccessor(agent_index, action)
            new_value = self.value(next_game_state, depth, next_agent_index)[0]
            v = min(v, new_value)
            if previous_v != v:
                best_action = action
        return [v, best_action]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        return self.value(gameState, 0, 0, -math.inf, math.inf)[1]

    def value(self, game_state, depth, agent_index, alpha, beta):
        if not game_state.getLegalActions(
                agent_index) or depth == self.depth or game_state.isWin() or game_state.isLose():
            return [self.evaluationFunction(game_state), 0]

        # If the agent is the last ghost, increment the depth and set the next agent to pacman
        if agent_index == game_state.getNumAgents() - 1:
            depth += 1
            next_agent_index = 0
        # Otherwise, set the next agent to the next ghost
        else:
            next_agent_index = agent_index + 1

        if agent_index == 0:
            return self.max_value(game_state, depth, agent_index, next_agent_index, alpha, beta)
        else:
            return self.min_value(game_state, depth, agent_index, next_agent_index, alpha, beta)

    def max_value(self, game_state, depth, agent_index, next_agent_index, alpha, beta):
        v = -math.inf
        best_action = Directions.STOP
        for action in game_state.getLegalActions(agent_index):
            previous_v = v
            next_game_state = game_state.generateSuccessor(agent_index, action)
            new_value = self.value(next_game_state, depth, next_agent_index, alpha, beta)[0]
            v = max(v, new_value)
            if v > beta: return [v, best_action]

            alpha = max(alpha, v)
            if previous_v != v:
                best_action = action
        return [v, best_action]

    def min_value(self, game_state, depth, agent_index, next_agent_index, alpha, beta):
        v = math.inf
        best_action = Directions.STOP
        for action in game_state.getLegalActions(agent_index):
            previous_v = v
            next_game_state = game_state.generateSuccessor(agent_index, action)
            new_value = self.value(next_game_state, depth, next_agent_index, alpha, beta)[0]
            v = min(v, new_value)
            if v < alpha: return [v, best_action]

            beta = min(beta, v)
            if previous_v != v:
                best_action = action
        return [v, best_action]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.value(gameState, 0, 0)[1]

    def value(self, game_state, depth, agent_index):
        if not game_state.getLegalActions(
                agent_index) or depth == self.depth or game_state.isWin() or game_state.isLose():
            return [self.evaluationFunction(game_state), 0]

        # If the agent is the last ghost, increment the depth and set the next agent to pacman
        if agent_index == game_state.getNumAgents() - 1:
            depth += 1
            next_agent_index = 0
        # Otherwise, set the next agent to the next ghost
        else:
            next_agent_index = agent_index + 1

        if agent_index == 0:
            return self.max_value(game_state, depth, agent_index, next_agent_index)
        else:
            return self.exp_value(game_state, depth, agent_index, next_agent_index)

    def max_value(self, game_state, depth, agent_index, next_agent_index):
        v = -math.inf
        best_action = Directions.STOP
        for action in game_state.getLegalActions(agent_index):
            previous_v = v
            next_game_state = game_state.generateSuccessor(agent_index, action)
            new_value = self.value(next_game_state, depth, next_agent_index)[0]
            v = max(v, new_value)
            if previous_v != v:
                best_action = action
        return [v, best_action]

    def exp_value(self, game_state, depth, agent_index, next_agent_index):
        v = 0
        best_action = Directions.STOP
        for action in game_state.getLegalActions(agent_index):
            previous_v = v
            next_game_state = game_state.generateSuccessor(agent_index, action)
            new_value = self.value(next_game_state, depth, next_agent_index)[0]
            prob = self.probability(game_state, agent_index)
            v += prob * new_value
            if previous_v != v:
                best_action = action
        return [v, best_action]

    def probability(self, game_state, agent_index):
        return 1.0 / len(game_state.getLegalActions(agent_index))


def betterEvaluationFunction(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    food_list = current_game_state.getFood().asList()
    scare_orbs = current_game_state.getCapsules()
    ghost_states = current_game_state.getGhostStates()
    # print(ghost_states[0])
    # print(ghost_states[0].scaredTimer)
    score = 0

    score -= len(food_list) * 20
    score -= len(scare_orbs) * 45

    harmful_ghosts = [x for x in ghost_states if x.scaredTimer > 0]
    not_harmful_ghosts = [x for x in ghost_states if x not in harmful_ghosts]

    pacmman_pos = current_game_state.getPacmanPosition()
    # print(pacmman_pos)
    # print([x.getPosition() for x in not_harmful_ghosts])
    harmful_ghosts_dist = [mazeDistance(pacmman_pos, position_float_to_int(x.getPosition()), current_game_state) for x
                           in harmful_ghosts]
    not_harmful_ghosts_dist = [mazeDistance(pacmman_pos, position_float_to_int(x.getPosition()), current_game_state) for
                               x in
                               not_harmful_ghosts]
    for harmful_dist in harmful_ghosts_dist:
        if harmful_dist < 5:
            score -= 30 * (5 - harmful_dist)

    for scared_dist in not_harmful_ghosts_dist:
        if scared_dist < 3:
            score += 30 * (3 - scared_dist)
        elif scared_dist < 6:
            score += 15 * (6 - scared_dist)
        else:
            score += 5

    # foods_dist = [mazeDistance(pacmman_pos, food_pos, current_game_state) for food_pos in food_list]

    # for food_dist in foods_dist:
    #     if food_dist < 3:
    #         score += 10
    #     elif food_dist < 6:
    #         score += 5
    return score


def position_float_to_int(position):
    return (int(position[0]), int(position[1]))


# Abbreviation
better = betterEvaluationFunction