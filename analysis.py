# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0005 # Reduce the noise as much as possible in order to reduce the chance of large negative discount making it more likely to move across bridge
    return answerDiscount, answerNoise

def question3a(): # for close exit, discount needs to be high in order make extra steps unfeasible, to risk the cliff the noise needs to be low in order to reduce chance of get -10 reward.
    answerDiscount = 0.3 # Trial and error from starting with 0
    answerNoise = 0.0005 # Used the value I calculated in q2 as a start point, it worked
    answerLivingReward = 0 # Living reward and discount used to achieve same thing, chose to focus on discount to make problem simpler
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b(): # for close exit, discount needs to be high in order make extra steps unfeasible, to avoid the cliff the noise needs to be high in order to increase chance of get -10 reward and ultimatly deter from taking chance
    answerDiscount = 0.3 # Used same value from Q 3A
    answerNoise = 0.2 # Used .2 as that was the noise value used in lectures
    answerLivingReward = 0 # Living reward and discount used to achieve same thing, chose to focus on discount to make problem simpler
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c(): # for far exit, discount needs to be low in order make extra steps feasible, to risk the cliff the noise needs to be low in order to reduce chance of get -10 reward.
    answerDiscount = 0.99 # Started with 1 and through trial and error began incrementlly decreasing
    answerNoise = 0.0005  # Used the value I calculated in q2 as a start point, it worked
    answerLivingReward = 0 # Living reward and discount used to achieve same thing, chose to focus on discount to make problem simpler
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d(): # for far exit, discount needs to be low in order make extra steps feasible, to avoid the cliff the noise needs to be high in order to increase chance of get -10 reward and ultimatly deter from taking chance
    answerDiscount = 0.99 # Used value from Q 3C
    answerNoise = 0.2 # Used vale from Q 3B
    answerLivingReward = 0 # Living reward and discount used to achieve same thing, chose to focus on discount to make problem simpler
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    #I achieved these values through trial and error
    answerDiscount = 0.4
    answerNoise = 0.4
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    #return answerEpsilon, answerLearningRate
    # If not possible,
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
