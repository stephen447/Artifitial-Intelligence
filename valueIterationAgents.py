# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100): # initialising mdp - iterations, discount, values dictionary and calls run value iteration
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range (self.iterations): # For the total number of iterations
            values=self.values.copy() # Creates a copy of the values for this iteration
            for states in self.mdp.getStates(): # for every state in the MDP - value can be calculated
                self.values[states] = -float('inf') # Let the initial value of the state be - infinity for max arg function to work properly
                for actions in self.mdp.getPossibleActions(states):  # For every possible action for the state in question
                    state_val=0 # state val is a variable for storing the current value of the state for a specific action within that state
                    for next_state,prob in self.mdp.getTransitionStatesAndProbs(states,actions): # For the action chosen finding the state transistion matrix which consists of state and probabilty of that state
                        state_val+=prob*(self.mdp.getReward(states, actions, next_state)+ self.discount *values[next_state]) # Bellman equation - prob of state
                    self.values[states] = max(self.values[states], state_val)
                if self.values[states] == -float('inf'): # If max is - infinity e.g. first round when iteration is zero, let the value equal 0, couldnt use zero initialyy as the q values could be negative and thus 0 would stay as max incorrectly
                    self.values[states] = 0.0



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_val=0 # Initialise the q value to 0
        
        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action): # For all possible next states for a specified action
            q_val += probability * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]) # Define the q value for the specified action withing a state
            
        return  q_val # return the q value of the state for an action
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state): # If it in the terminal state, no action required return nothing
            return None
            
        max_val, action = -float('inf'), None # Set initial value of the state to - infinity and the optimal action to none
        
        for actions in self.mdp.getPossibleActions(state): # For every possible action for the state
        
            curr_val = self.computeQValueFromValues(state, actions) # compute the q value for the state and action
            if curr_val > max_val:  # if the q value is greater than the current max q value
                max_val, action = curr_val, actions # Update the max value to the current value and the similiarly for the action
                
        return action # Return the optimal action for the state

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    

    def runValueIteration(self):

        for iter in range(self.iterations): # For loop for number of iterations specified

            state = self.mdp.getStates()[iter %  len(self.mdp.getStates())] # Cycling throguh states for each iteration using modulo
            optimal_action = self.computeActionFromValues(state) #Compute best action from current state
            
            if optimal_action is None: # If the best action is none, let value of state be zero
                state_val = 0
            else: # else let the value of the state equal the q value
                state_val = self.computeQValueFromValues(state, optimal_action)
                
            self.values[state] = state_val


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def q_vals(self, state): # Function for calculating the q value of a state
    

        actions = self.mdp.getPossibleActions(state)  # All possible actions from a state
        qValues = util.Counter()  # A counter holding (action, qValue) pairs - dictionary with all entries initialised to zero

        for action in actions:
            # Putting the calculated Q value for the given action into my counter
            qValues[action] = self.computeQValueFromValues(state, action)

        return qValues
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
    def runValueIteration(self):
        queue = util.PriorityQueue() # Empty queue for next states to be visited
        states = self.mdp.getStates() # get the states in the MDP
        previous_states = dict() # dictionary for storing previous states
        
        for current_state in states: # Cycling through each state
            previous_states[current_state]=set() # Add the current state to visited states
            
        for current_state in states: # Cycling through states
            actions=self.mdp.getPossibleActions(current_state) # Get every action for the current state
            for i in actions: # For every action calculate the next state and corresponding transition probability
                possibleNextStates = self.mdp.getTransitionStatesAndProbs(current_state, i)
                for nextState,probability in possibleNextStates: # For every next state for action
                    if probability>0: # If it is possible for state to be transistioned to from current state
                        previous_states[nextState].add(current_state) # Add the currrent state as the previous state for the next state
        
        for current_state in states: # For every state
            q_values = self.q_vals(current_state) # Find the  q values for the state

            if len(q_values) > 0: # If there is q values
                optimum_q = q_values[q_values.argMax()] # find the optimum q value
                diff = abs(self.values[current_state] - optimum_q) # Find the absolute  difference between the optimum q value and the rest of the q values
                queue.push(current_state, -diff) # put the neg difference value on the queue
                
        for i in range(self.iterations): # cycling through total number of iterations
            if queue.isEmpty(): # if the queue is empty, return
                return;
            current_state = queue.pop() # pop the current state of the queue so its not revisited next round
            q_values = self.q_vals(current_state) # compute the q values for the state
            optimum_q = q_values[q_values.argMax()] # find the maximum q value
            self.values[current_state] = optimum_q # store optimum q value
            for predecessor in previous_states[current_state]: # for each predecessor of current state

                predecessor_q_values = self.q_vals(predecessor) # calculate the q values of the predecessor
                optimum_predessor_q = predecessor_q_values[predecessor_q_values.argMax()] # find the optimum value of the predecessor
                q_difference = abs(self.values[predecessor] - optimum_predessor_q) # find the absolute value of the difference between optimum q value and the other q values

                if q_difference > self.theta: # if the difference is bigger than a certain value  theta, push it on the priority queue with a priority value of -q_difference(least priority to larger values)
                    queue.update(predecessor, -q_difference)

