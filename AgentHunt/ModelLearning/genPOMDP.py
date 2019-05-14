from utils import getDifficulties, calcAccuracy
from itertools import product
############################
# There are (numDiffs) * (numDiffs) * 2 states +  1 terminal state at the end.
# We index as follows: Suppose Type A is the 0th difficulty, 
# type B is the 5th difficulty, and the answer is zero.
# Then, the corresponding state number is 
# (0 * numDiffs * numDiffs) + (0 * numDiffs) + 5.
#
# Essentially, the first half of the states represents answer zero
# The second half represents answer one
# Each half is divided into numDiffs sections, representing 
# each possible difficulty for a typeA question.
# Then each section is divided into numDiffs sections, representing
# each possible difficulty for a typeB question.
###########################

def genPOMDP(filename, reward, cost, gammas, numberOfWorkflows):    
    difficulties = getDifficulties(0.1)
    numDiffs = len(difficulties)
    
    #Add one absorbing state
    numberOfStates = ((numDiffs) ** numberOfWorkflows * 2) + 1
    numberOfActions = numberOfWorkflows + 2
    file = open(filename, 'w')
    file.write('discount: 0.9999\n')
    file.write('values: reward\n')
    file.write('states: %d\n' % numberOfStates)
    file.write('actions: %d\n' % numberOfActions)
    SUBMITZERO = numberOfWorkflows
    SUBMITONE = numberOfWorkflows + 1
    file.write('observations: Zero One None\n')

    for i in range(0, numberOfStates):
        for k in range(0, numberOfWorkflows):
            file.write('T: %d : %d : %d %f\n' % (k, i, i, 1.0))

    #Add transitions to absorbing state
    file.write('T: %d : * : %d %f\n' % (SUBMITZERO, numberOfStates-1, 1.0))
    file.write('T: %d : * : %d %f\n' % (SUBMITONE, numberOfStates-1, 1.0))

    #Add observations in absorbing state
    file.write('O: * : %d : None %f\n' % (numberOfStates-1, 1.0))

    for v in range(0, 2):
        for diffState in product(range(numDiffs), repeat = numberOfWorkflows):
            state = v * (numDiffs ** numberOfWorkflows)
            for k in range(0, numberOfWorkflows):
                state += (diffState[k] * (numDiffs ** (numberOfWorkflows - (k+1))))
            file.write('O: %d: %d : None %f\n' % (SUBMITZERO, state, 1.0))
            file.write('O: %d: %d : None %f\n' % (SUBMITONE, state, 1.0))
            if v == 0: #if the answer is 0
                for k in range(0, numberOfWorkflows):
                    file.write('O: %d : %d : Zero %f\n' % 
                               (k, state, 
                                calcAccuracy(gammas[k], difficulties[diffState[k]])))
                    file.write('O: %d : %d : One %f\n' % 
                               (k, state, 1.0 - calcAccuracy(gammas[k], 
                                                             difficulties[diffState[k]])))
            else: # if the answer is 1
                for k in range(0, numberOfWorkflows):
                    file.write('O: %d : %d : Zero %f\n' % 
                               (k, state, 
                                1.0 - calcAccuracy(gammas[k], difficulties[diffState[k]])))
                    file.write('O: %d : %d : One %f\n' % 
                               (k, state, calcAccuracy(gammas[k], 
                                                       difficulties[diffState[k]])))
    
    file.write('R: * : * : * : * %f\n' % (-1 * cost))


    for i in range(0, numberOfStates-1):
        if i < (numberOfStates-1) / 2:
            file.write('R: %d : %d : %d : * %f\n' % (SUBMITZERO, i, numberOfStates-1, 1))
            file.write('R: %d : %d : %d : * %f\n' % (SUBMITONE, i, numberOfStates-1, reward))
        else:
            file.write('R: %d : %d : %d : * %f\n' % (SUBMITONE, i, numberOfStates-1, 1))
            file.write('R: %d : %d : %d : * %f\n' % (SUBMITZERO, i, numberOfStates-1, reward))

    #Add rewards in absorbing state
    file.write('R: * : %d : %d : * %f\n' % (numberOfStates-1, numberOfStates-1, 0))

    file.close()
