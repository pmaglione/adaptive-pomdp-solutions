from ModelLearning.utils import *
from ModelLearning.genPOMDP import *
from Data import *
from Ballots import *

import time
import subprocess
from random import random
from os import mkdir, rmdir
from copy import deepcopy

#Statuses
DONE = 2
WAITING_FOR_TURKER = 1
READY_FOR_ACTION = 0

def numberLeft(statuses):
    count = 0
    for status in statuses:
        if status != DONE:
            count += 1
    return count

def solve(mt, numStates, numberOfProblems, nameOfTask, value, price, 
          ZMDPPATH, URL, EMPATH, fastLearning, timeLearning, 
          taskDuration, debug = False):

    #Initialize array of difficulties
    difficulties = getDifficulties(0.1)

    BALLOT0 = 0 
    BALLOT1 = 1 
    SUBMITZERO = 2
    SUBMITONE = 3

    choices = [{} for i in range(0, numberOfProblems)]
    inversechoices = [{} for i in range(0, numberOfProblems)]
    choicesindex = [0 for i in range(0, numberOfProblems)]

    print "Initializing World State"

    #Initialize status, answers, costs, HITIds, userWorkers, available actions,
    #actions that the agent took
    statuses = [READY_FOR_ACTION for i in range(0, numberOfProblems)]
    answers = [-1 for i in range(0, numberOfProblems)]
    costs = [0 for i in range(0, numberOfProblems)]
    HITIds = []
    usedWorkers = [[] for i in range(0, numberOfProblems)]
    actions = range(0, 4)
    agentActions = [-1 for i in range(0, numberOfProblems)]    
    ballots = Ballots(EMPATH)

    #We should start out with uniform beliefs
    belief = normalize([1 for i in range(numStates)])
    beliefs = [deepcopy(belief) for i in range(numberOfProblems)]
    #print beliefs[0]

    serialize(numberOfProblems, statuses, answers, costs, 
              HITIds, usedWorkers, actions, agentActions,
              beliefs, ballots)

    #We keep the observations around for RL purposes
    observations0 = [[] for i in range(0, numberOfProblems)] 
    observations1 = [[] for i in range(0, numberOfProblems)]

    (gamma0, gamma1) = ballots.calcAverageGammas()

    print "Reading Policy"

    ###########################################################
    #Read the policy that we will begin with
    #You can choose a policy that's already been learned or learn
    #a new one
    ###########################################################
    genPOMDP('log/pomdp/rl.pomdp', 
             value * -1, 1, 
             [1.0, 1.0],
             2)
    #Solve the POMDP
    zmdpDumpfile = open('log/pomdp/zmdpDump', 'w')
    subprocess.call('%s solve %s -o %s -t %d' % (
            ZMDPPATH,
            'log/pomdp/rl.pomdp',
            'log/pomdp/out.policy',
            timeLearning),
                    stdout=zmdpDumpfile,
                    shell=True,
                    executable="/bin/bash")
    zmdpDumpfile.close()
                        #Read the policy that we will begin with 
    policy = readPolicy("log/pomdp/out.policy", 
                        numStates)
    ################################################################
    # If you want a policy that we've already learned
    # pick one from ModelLearning/Policies.
    # W2R100 means 2 workflows, value of answer is 100
    #################################################################
    #policy = readPolicy("ModelLearning/Policies/W2R100.policy", 
    #                    numStates)

    print "POMDP Generated, POMDP Solved, Policy read"

    print "Initialization Complete"
    
    #Begin the experiment
    iterationNumber = -1
    while numberLeft(statuses) != 0:
        iterationNumber += 1
        availableQuestions0 = []
        availableQuestions1 = []
        if debug:
            print "Waiting 10 seconds"
        time.sleep(30)
        print "Beginning iteration %d with %d questions remaining" % (
            iterationNumber, numberLeft(statuses))
        #First we read from file all our data.
        (nop, statuses, answers, costs, HITIds, 
         usedWorkers, actions, agentActions, beliefs, ballots) = unSerialize()
        for i in range(0, numberOfProblems):
            if debug:
                print "Starting Problem %d:" % i
            if statuses[i] == DONE:
                if debug:
                    print "Problem %d is done" % i
                continue
            elif statuses[i] == READY_FOR_ACTION:
                beliefState = beliefs[i]
                bestAction = findBestAction(actions, policy, beliefState)
                agentActions[i] = bestAction
                #Exploration versus exploitation
                if random() <= 0.1:
                    if bestAction == BALLOT0:
                        bestAction = BALLOT1
                    elif bestAction == BALLOT1:
                        bestAction = BALLOT0
                    else:
                        if random() <= 0.5:
                            bestAction = BALLOT0
                        else:
                            bestAction = BALLOT1
                if bestAction == BALLOT0 or bestAction == BALLOT1:
                    if bestAction == BALLOT0:
                        if debug:
                            print "Problem %d requested workflow %d" % (i, 
                                                                        BALLOT0)
                        availableQuestions0.append(i)
                    elif bestAction == BALLOT1:
                        if debug:
                            print "Problem %d requested workflow %d" % (i,
                                                                        BALLOT1)
                        availableQuestions1.append(i)

                    costs[i] += 1
                    statuses[i] = WAITING_FOR_TURKER
                elif bestAction == SUBMITZERO or bestAction == SUBMITONE:
                    ballots.addQuestionAndRelearn(
                        observations0[i],
                        observations1[i],
                        bestAction - 2,
                        getMostLikelyDifficulties(beliefs[i], difficulties),
                        fastLearning)
                    (gamma0, gamma1) = ballots.calcAverageGammas()
                    if not fastLearning:
                        print "Problem %d complete: Relearning." % i
                        #Generate the POMDP
                        genPOMDP('log/pomdp/rl.pomdp', 
                                 value * -1, 1, 
                                 [gamma0, gamma1],
                                 2)
                        #Solve the POMDP
                        zmdpDumpfile = open('log/pomdp/zmdpDump', 'w')
                        subprocess.call('%s solve %s -o %s -t %d' % (
                                ZMDPPATH,
                                'log/pomdp/rl.pomdp',
                                'log/pomdp/out.policy',
                                timeLearning),
                                        stdout=zmdpDumpfile,
                                        shell=True,
                                        executable="/bin/bash")
                        zmdpDumpfile.close()
                        #Read the policy that we will begin with 
                        policy = readPolicy("log/pomdp/out.policy", 
                                            numStates)

                        print "POMDP Generated, POMDP Solved, Policy read"
                    if bestAction == SUBMITZERO:
                        if debug:
                            print "Problem %d submitted answer 0" % i 
                        answers[i] = 0
                    else:
                        if debug:
                            print "Problem %d submitted answer 1" % i
                        answers[i] = 1
                    statuses[i] = DONE
            elif statuses[i] == WAITING_FOR_TURKER:
                continue
            
        #Push out all the hits
        #First get Locks
        while True:
            try:
                mkdir('locks/aql0lock')
                break
            except OSError:
                pass
        while True:
            try:
                mkdir('locks/aql1lock')
                break
            except OSError:
                pass

        #We only append to these files.
        aql0 = open('log/aql0', 'a')
        aql1 = open('log/aql1', 'a')
        
        for aq in availableQuestions0:
            aql0.write('%d,' % aq)
            HITIds.append(mt.createHIT(0, URL, nameOfTask, price, taskDuration,
                                       31536000,1,aq))
        for aq in availableQuestions1:
            aql1.write('%d,' % aq)
            HITIds.append(mt.createHIT(1, URL, nameOfTask, price, taskDuration,
                                       31536000,1,aq))

                          
        #Release Locks
        rmdir('locks/aql0lock')
        rmdir('locks/aql1lock')

        #Now we go through all the hits and get observations
        nextHITIds = []
        for HITId in HITIds: 
            hits = mt.getHIT(HITId)
            observation = ''
            for hit in hits: #there really should only be one
                if hit.HITStatus == 'Reviewable':
                    assignments = mt.getAssignments(HITId, 
                                                    page_size=100,
                                                    page_number=1)
                    for assignment in assignments: #there should only be one
                        observation = mt.getObservation(assignment)
                        pn = mt.getProblemNumber(assignment)
                        print "THE OBSERVATION"
                        print observation

                        if observation in choices[pn]:
                            observation = choices[pn][observation]
                        else:
                            inversechoices[pn][choicesindex[pn]] = observation
                            choices[pn][observation] = choicesindex[pn]
                            choicesindex[pn] += 1
                            observation = choices[pn][observation]
                        
                        workerId = mt.getWorker(assignment)
                        mt.approveAssignment(assignment)

                        if debug:
                            print "Problem %d received observation %d" % (pn, observation)
                        #Now that we have an observation, we need to update our belief
                    beliefs[pn] = updateBelief(beliefs[pn], agentActions[pn],
                                              observation, difficulties,
                                              ballots.getWorkerGamma0(workerId),
                                              ballots.getWorkerGamma1(workerId))
                    if agentActions[pn] == 0:
                        f = open('log/results/observations%dw0' % pn, 'a+')
                        observations0[pn].append((observation, workerId))
                    else:
                        f = open('log/results/observations%dw1' % pn, 'a+')
                        observations1[pn].append((observation, workerId))
                    f.write('%s\t%s\n' % (inversechoices[pn][observation], 
                                        workerId))
                    f.close()
                                                  
                    mt.disposeHIT(HITId) 
                    statuses[pn] = READY_FOR_ACTION
                else:
                    nextHITIds.append(HITId)
        HITIds = nextHITIds

        #Write our world state to file in case something crashes and burns
        serialize(numberOfProblems, statuses, answers, 
                  costs, HITIds, usedWorkers,
                  actions, agentActions, beliefs, ballots)

    f = open('log/results/answers', 'w') 
    for i in range(numberOfProblems):
        f.write('%s\n' % inversechoices[i][answers[i]]) 
    return (costs, answers)
