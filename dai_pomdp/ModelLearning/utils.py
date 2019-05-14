from random import random
from math import sqrt
from itertools import product
import numpy as np


def calcAccuracy(gamma, d):
    return (1.0 / 2) * (1.0 + (1.0 - d) ** gamma)


def calcBallotProb(ballots, answer, d):
    totalProb = 1.0
    for i in range(0, len(ballots)):
        prob = 0.0
        if ballots[i] == answer:
            prob = calcAccuracy(1, d)
        else:
            prob = (1 - calcAccuracy(1, d))
        totalProb *= prob
    return totalProb


def normalize(array):
    sum = 0.0
    for i in range(0, len(array)):
        sum += array[i]  # when initialize all in 1, equivalent to 1 * numStates
    for i in range(0, len(array)):
        array[i] = array[i] / sum  # when initialize all in 1, equivalent to 1 / numStates
    return array


def dot(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        if v1[i] == '*' or v2[i] == '*':
            continue
        sum += v1[i] * v2[i]
    return sum

def generateBallot(gamma, d, answer):
    a = random()
    if a < calcAccuracy(gamma, d):
        return answer
    else:
        return 1 - answer

def generateBallotWithAcc(accuracy, answer):
    if np.random.binomial(1, accuracy):
        return answer
    else:
        return 1 - answer


# Generate discrete difficulties between 0 and 1 at diffInterval steps
def getDifficulties(diffInterval):
    difficulties = []
    numDiffs = int(1.0 / diffInterval + 1)
    for i in range(0, numDiffs):
        difficulties.append(round(diffInterval * i, 1))
    return difficulties


def findBestAction(actions, policy, beliefState):
    bestValue = -1230981239102938019
    bestAction = 0  #Assume there is at least one action
    for action in actions:
        if action in policy:
            value = findBestValue(action, policy[action], beliefState)
            if value > bestValue:
                bestValue = value
                bestAction = action
    return bestAction


def findBestValue(action, hyperplanes, beliefs):
    bestValue = -129837198273981231
    for hyperplane in hyperplanes:
        dontUse = False
        for (b, entry) in zip(beliefs, hyperplane):
            if b != 0 and entry == '*':
                dontUse = True
                break
        if dontUse:
            continue
        value = dot(beliefs, hyperplane)
        if value > bestValue:
            bestValue = value

    return bestValue


def getMostLikelyDifficulty(belief, difficulties):
    numDiffs = len(difficulties)
    bestState = -1
    bestProb = 0
    for i in range(0, 2):
        for j in range(0, numDiffs):
            diff = difficulties[j]
            state = i * numDiffs + j
            if belief[state] > bestProb:
                bestState = diff
                bestProb = belief[state]
    return bestState


def updateBelief(prevBelief, observation, difficulties, gamma):
    newBeliefs = []
    numDiffs = len(difficulties)
    for i in range(0, 2):  # 0,1
        for j in range(0, numDiffs):  # 11
            diff = difficulties[j]
            state = (i * numDiffs) + j

            if observation == i:
                newBeliefs.append(calcAccuracy(gamma, diff) * prevBelief[state])
            else:
                newBeliefs.append((1 - calcAccuracy(gamma, diff)) * prevBelief[state])

    newBeliefs.append(0.0)
    normalize(newBeliefs)
    return newBeliefs


def readPolicy(policyfile, numStates):
    try:
        policy = {}
        fs = open(policyfile, 'r')
        lines = fs.read().split("\n")

        numPlanes = 0
        action = 0
        alpha = [0 for k in range(0, numStates)]
        insideEntries = False
        for i in range(0, len(lines)):
            line = lines[i]
            #First we ignore a bunch of lines at the beginning
            if (line.find('#') != -1 or line.find('{') != -1 or
                        line.find('policyType') != -1 or line.find('}') != -1 or
                        line.find('numPlanes') != -1 or
                    ((line.find(']') != -1) and not insideEntries) or
                        line.find('planes') != -1 or line == ''):
                continue
            if line.find('action') != -1:
                words = line.strip(', ').split(" => ")
                action = int(words[1])
                continue
            if line.find('numEntries') != -1:
                continue
            if line.find('entries') != -1:
                insideEntries = True
                continue
            if (line.find(']') != -1) and insideEntries:  #We are done with one alpha vector
                if action not in policy:
                    policy[action] = []
                policy[action].append(alpha)
                action = 0
                alpha = ['*' for k in range(0, numStates)]
                numPlanes += 1
                insideEntries = False
                continue
            #If we get here, we are reading state value pairs
            entry = line.split(",")
            state = int(entry[0])
            val = float(entry[1])
            alpha[state] = val

        return policy
    except:
        raise FileNotFoundError



