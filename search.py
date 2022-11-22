# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    start_state = problem.getStartState() # Find the node which we are starting with
    if problem.isGoalState(start_state): # If we are starting with the goal, return an empty list - since there were no actions needed to get to the goal
        return []
    
    fringe = util.Stack() # Using LIFO for depth first search, creating stack for the fringe which is for the nodes which need to be visited
    fringe.push((start_state, []))  # Push the starting node and the list of actions up to now onto the fringe
    visited = []  # List for the nodes which were visited to prevent them being revisited

    while not fringe.isEmpty(): # While there is still nodes to be visited in the fringe
        cur_node, actions = fringe.pop()  # Assign the last entry from the fringe to the current node and current actions
        if cur_node not in visited:  # If the current node hasnt been visited
            visited.append(cur_node)  # Add the current node to the set of visited nodes
            if problem.isGoalState(cur_node):  # If the goal state has been reached can return the set of actions
                return actions
            for sucessor_node, action, cost in problem.getSuccessors(cur_node): # For each of the neighbours of the current node , push the neighbours, their corresponding action and cost to the fringe
                sucessor_action = actions + [action] # Add the actions up to now and the actions to get to the sucessor
                fringe.push((sucessor_node, sucessor_action))  # Push the data to the fringe
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    #print("Start:", problem.getStartState()) # Find the node which we are starting with
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()  # Get the start state for the problem
    visited = []  # List for the nodes which were visited to prevent them being revisited
    visited.append(start_state)  # add the start state to vistied
    fringe = util.Queue()  # The fringe is now a queue for BFS, which operates with FIFO
    fr = (start_state, [])
    fringe.push(fr)  #  Add the start state to the fringe, so its sucessors can be calculated
    while not fringe.isEmpty():  #  while the fringe is not empty, keep searching
        cur_node, action = fringe.pop()  # Load in next entry in queue to the current node
        if problem.isGoalState(cur_node):  # If goal state, the search is complete and return action
            return action
        successor = problem.getSuccessors(cur_node)  # get the sucessor of the current node
        for suc in successor:  #  for every sucessor pf current node
            node = suc[0]  # the node is in position 0 of suc
            if not node in visited:  #  if the node hasnt been visited
                act = suc[1] # get the action for the sucessor from index 1
                visited.append(node)  # add the node to explored states
                fringe.push((node, action + [act]))
    return action
    #util.raiseNotDefined()
     
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState() # Find the node which we are starting with
    
    if problem.isGoalState(start_state): # If we are starting with the goal, return an empty list - since there were no actions needed to get to the goal
        return []
    
    fringe = util.PriorityQueue()  # Fringe for UCS is a queue which is ordered from lowest cost to highest cost
    fringe.push((start_state, [], 0), 0)  # Push the the starting node on to the priority queue with corresponding actions and cost of zero since its the start node
    visited = [] # List for the nodes which were visited to prevent them being revisited
    
    while not fringe.isEmpty(): # While there is still nodes to be visited in the fringe
        cur_node, actions, last_cost = fringe.pop() # Assign the top of the fringe to corresponding current node, actions and the cost
        if cur_node not in visited:   # If the current node hasnt been visited
            visited.append(cur_node)  # Add the current node to the set of visited nodes
            if problem.isGoalState(cur_node):   # If the goal state has been reached, can return the set of actions
                return actions
            for sucessor_node, action, cost in problem.getSuccessors(cur_node): # for every neighbour of the current node load in its state, actions and cost
                sucessor_action = actions+[action] # Update the actions for the node in question by adding the action to list of actions
                priority = last_cost+cost # Update the cost for the node in question
                fringe.push((sucessor_node, sucessor_action, priority), priority) # Push the paramters on to the priority queue

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start_node = problem.getStartState()  # Find the node which we are starting with
    visited = [] # List for the nodes which were visited to prevent them being revisited
    fringe = util.PriorityQueue()  # Fringe for UCS is a queue which is ordered from lowest cost to highest cost
        
    if problem.isGoalState(start_node):  # If we are starting with the goal, return an empty list - since there were no actions needed to get to the goal
        return []

    fringe.push((start_node, [], 0), 0) # Push the the starting node on to the priority queue with corresponding actions and cost of zero since its the start node
    
    
    while not fringe.isEmpty(): # While there is still nodes to be visited in the fringe
        cur_node, actions, last_cost = fringe.pop() # Load in the next values of node, the action to get there and cost
        if cur_node not in visited: # If the current node hasnt been visited
            visited.append(cur_node) # Add the current node to the set of visited nodes
            if problem.isGoalState(cur_node):
                return actions
            for sucessor_node, action, cost in problem.getSuccessors(cur_node):
                sucessor_action = actions + [action] # Update the list of actions for the sucessor
                updated_cost = last_cost+cost  # Update the cost for the sucessor using the old cost to get to the node and then the cost from the node to the sucessor
                estimated_cost = updated_cost+heuristic(sucessor_node, problem) # Update the estimated cost, by added the updated cost to the sucessor and the heuristic cost from the sucessor to the goal
                fringe.push((sucessor_node, sucessor_action, updated_cost), estimated_cost) # Push the sucessor, actions to get to the sucessor and the cost to get to the sucessor to the priority queue so it can be visited

    #util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
