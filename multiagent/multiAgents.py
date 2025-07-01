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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood() # food[x][y]=True
        newGhostStates = successorGameState.getGhostStates() # list
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodm = 1
        foodList = newFood.asList()
        if len(foodList) > 0: 
            foodm = 99999
            for food in foodList:
                foodm = min(foodm, util.manhattanDistance(newPos, food))
        if_zero = False
        if foodm == 0 or foodm == 1: 
            if foodm == 0: if_zero = True
            foodm = 1
        foodBonus = 10000.0 - foodm
        if if_zero: foodBonus += 10000
        foodPanelty = len(foodList)
        ghostPanelty = 0
        ghostBonus = 0
        ghost_num = len(newGhostStates)
        for i in range(0, ghost_num):
            distance = util.manhattanDistance(newPos, newGhostStates[i].getPosition())
            if newScaredTimes[i] == 0: 
                if distance == 0: distance = 1
                ghostPanelty += 1000.0 - 0.5 * distance
                if distance == 1: ghostPanelty += 10000
            else: 
                if distance == 0: distance = 1
                ghostBonus += 100.0 - 3 * distance
                if distance == 1: ghostBonus += 1000

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore() - ghostPanelty + ghostBonus + foodBonus - foodPanelty * 10

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState:GameState, depth, agentIndex):
        agentNum = gameState.getNumAgents()
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0: # 是Pacman
            maxEval = -999999
            for action in gameState.getLegalActions(agentIndex):
                new_state = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex + 1) == agentNum: dep = depth - 1
                else: dep = depth
                newEval = self.minimax(new_state, dep, (agentIndex + 1) % agentNum)
                maxEval = max(maxEval, newEval)
            return maxEval
        else:
            minEval = 999999
            for action in gameState.getLegalActions(agentIndex):
                new_state = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex + 1) == agentNum: dep = depth - 1
                else: dep = depth
                newEval = self.minimax(new_state, dep, (agentIndex + 1) % agentNum)
                minEval = min(minEval, newEval)
            return minEval


    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"

        bestScore = -99999
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.minimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = -99999
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                if nextAgent == 0:
                    newDepth = depth - 1
                    value = max_value(successor, newDepth, nextAgent, alpha, beta)
                else:
                    newDepth = depth
                    value = min_value(successor, newDepth, nextAgent, alpha, beta)
                v = max(v, value)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = 99999
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                if nextAgent == 0:
                    newDepth = depth - 1
                    value = max_value(successor, newDepth, nextAgent, alpha, beta)
                else:
                    newDepth = depth
                    value = min_value(successor, newDepth, nextAgent, alpha, beta)
                v = min(v, value)
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        bestAction = None
        bestValue = -99999
        alpha = -99999
        beta = 99999

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = min_value(successor, self.depth, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = -99999
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                if nextAgent == 0:
                    newDepth = depth - 1
                    value = max_value(successor, newDepth, nextAgent)
                else:
                    newDepth = depth
                    value = min_value(successor, newDepth, nextAgent)
                v = max(v, value)
            return v

        def min_value(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = 0
            action_num = float(len(state.getLegalActions(agentIndex)))
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                if nextAgent == 0:
                    newDepth = depth - 1
                    value = max_value(successor, newDepth, nextAgent)
                else:
                    newDepth = depth
                    value = min_value(successor, newDepth, nextAgent)
                v += value
            return v / action_num

        bestAction = None
        bestValue = -99999

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = min_value(successor, self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
    
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #foodGrid = currentGameState.getFood()
    #pos = currentGameState.getPacmanPosition()
    #walls = currentGameState.getWalls()
    #if walls[pos[0]][pos[1]]: return 0
    #if currentGameState.getNumFood() == 0: return 0
    #def getNearestFood(startState: GameState):
    #    foodGrid = currentGameState.getFood()
    #    pos = currentGameState.getPacmanPosition()
    #    q = util.Queue()
    #    q.push(startState)
    #    visited = set()
    #    length = 0
    #
    #    while not q.isEmpty():
    #        state = q.pop()
    #
    #        if state in visited:
    #            continue
    #        visited.add(state)
#
    #        pos1 = state.getPacmanPosition()
    #        if foodGrid[pos1[0]][pos1[1]]:
    #            return length
    #
    #        for action1 in startState.getLegalPacmanActions():
    #            suc_state = startState.generatePacmanSuccessor(action1)
    #            successor = suc_state
    #            if successor not in visited:
    #                length += 1
    #                q.push(successor)
    #    return length
    #def getGhostDis(startState: GameState, ghostIndex: int):
    #    ghosts = currentGameState.getGhostPositions()
    #    pos = currentGameState.getPacmanPosition()
    #    q = util.Queue()
    #    q.push(startState)
    #    visited = set()
    #    length = 0
    #
    #    while not q.isEmpty():
    #        state = q.pop()
    #
    #        if state in visited:
    #            continue
    #        visited.add(state)
#
    #        pos1 = state.getPacmanPosition()
#
    #        if pos1 == ghosts[ghostIndex]:
    #            return length
    #
    #        for action1 in startState.getLegalPacmanActions():
    #            suc_state = startState.generatePacmanSuccessor(action1)
    #            successor = suc_state
    #            if successor not in visited:
    #                length += 1
    #                q.push(successor)
    #    return length
    #foodBonus = 10000 - getNearestFood(currentGameState)
#
    #
    #newGhostStates = currentGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #
    #ghostPenalty = 0
    #ghostBonus = 0
#
    #ghost_num = len(newGhostStates)
    #for i in range(0, ghost_num):
    #    if newScaredTimes[i] == 0: 
    #        dis = getGhostDis(currentGameState, i)
    #        ghostPenalty -= getGhostDis(currentGameState, i)
    #        if dis == 1: ghostPenalty -= 100
    #    else: 
    #        ghostBonus += getGhostDis(currentGameState, i)
    #
    #def surrounded(pos):
    #    if walls[pos[0]][pos[1]]: return 3
    #    num = 0
    #    if walls[pos[0] + 1][pos[1]]: num += 1
    #    if walls[pos[0] - 1][pos[1]]: num += 1
    #    if walls[pos[0]][pos[1] + 1]: num += 1
    #    if walls[pos[0]][pos[1] - 1]: num += 1
    #    return num
    #
    #foodScore = 0
    #if foodGrid[pos[0]][pos[1]]: foodScore = 10
    #
    #SurroundPenalty = 0
    #if surrounded(pos) == 3 or surrounded(pos) == 4: 
    #    if foodGrid[pos[0]][pos[1]] == False: SurroundPenalty = -1000
    #elif surrounded(pos) == 2 and foodGrid[pos[0]][pos[1]] == False: SurroundPenalty = -10
#
    #foodList = foodGrid.asList()
    #foodPanelty = -len(foodList)
#
    #return foodScore * 10 + foodBonus * 10 + SurroundPenalty + currentGameState.getScore() * 10 + foodPanelty
    #util.raiseNotDefined()

    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood() # food[x][y]=True
    newGhostStates = successorGameState.getGhostStates() # list
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodm = 1
    foodList = newFood.asList()
    if len(foodList) > 0: 
        foodm = 99999
        for food in foodList:
            foodm = min(foodm, util.manhattanDistance(newPos, food))
    if_zero = False
    if foodm == 0 or foodm == 1: 
        if foodm == 0: if_zero = True
        foodm = 1
    foodBonus = 10000.0 - foodm
    if if_zero: foodBonus += 10000
    foodPanelty = len(foodList)
    ghostPanelty = 0
    ghostBonus = 0
    ghost_num = len(newGhostStates)
    for i in range(0, ghost_num):
        distance = util.manhattanDistance(newPos, newGhostStates[i].getPosition())
        if newScaredTimes[i] == 0: 
            if distance == 0: distance = 1
            ghostPanelty += 1000.0 - 0.5 * distance
            if distance == 1: ghostPanelty += 10000
        else: 
            if distance == 0: distance = 1
            ghostBonus += 100.0 - 3 * distance
            if distance == 1: ghostBonus += 1000

    "*** YOUR CODE HERE ***"
    return successorGameState.getScore() - ghostPanelty + ghostBonus * 10 + foodBonus - foodPanelty * 10

# Abbreviation
better = betterEvaluationFunction
