from ModelLearning.utils import *
from ModelLearning.genPOMDP import *
from Data import *
from Ballots import *
from helpers import *

import time
import subprocess
from random import random
from os import mkdir, rmdir
from copy import deepcopy
from math import floor
from functools import reduce

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

def solve(mt, numStates, numberOfProblems, numberOfWorkerPools,
          nameOfTask, value, priceList, gammaList, scaleFactor,
          ZMDPPATH, URL, EMPATH, fastLearning, timeLearning, 
          taskDuration, debug = False, isLiveExperiment = False, expert_cost=20):


    #costList = map(lambda x: scaleFactor*x, priceList)
    costList = [scaleFactor * x for x in priceList]


    difficulties = getDifficulties(0.1)

    avg_error_rate = gammaList[0]


    # SINGLE WORKER POOL SETTING


    WORKERPOOLACTIONS = range(numberOfWorkerPools)
    SUBMITZERO = numberOfWorkerPools
    SUBMITONE = numberOfWorkerPools+1
    #UNCLASSIFIED = numberOfWorkerPools + 2  # for 1 pool == 3

    items_votes = {}
    for item_id in range(numberOfProblems):
        items_votes[item_id] = {}

    # 0 = +vote
    # 1 = submit zero
    # 2 = submit one
    # 3 = leave unclassified

    choices = [{'0':0,'1':1} for i in range(0, numberOfProblems)]
    inversechoices = [{0:'0',1:'1'} for i in range(0, numberOfProblems)]
    choicesindex = [0 for i in range(0, numberOfProblems)]

    print("Initializing World State")

    #Initialize status, answers, costs, HITIds, userWorkers, available actions,
    #actions that the agent took
    statuses = [READY_FOR_ACTION for i in range(0, numberOfProblems)]
    answers = [-1 for i in range(0, numberOfProblems)]
    costs = [0 for i in range(0, numberOfProblems)]
    HITIds = []
    usedWorkers = [[] for i in range(0, numberOfProblems)] #Should be a set
    actions = range(0, numberOfWorkerPools+2)
    agentActions = [-1 for i in range(0, numberOfProblems)]    
    ballots = Ballots(EMPATH,numberOfWorkerPools,gammaList)  # BALLOTS OBJECT

    #We should start out with uniform beliefs
    belief = [1 for i in range(numStates)]  # init , equivalent to [1] * numStates
    belief[numStates-1] = 0  # last states = 0, terminating state
    belief = normalize(belief)
    beliefs = [deepcopy(belief) for i in range(numberOfProblems)]
    #print beliefs[0]

    serialize(numberOfProblems, statuses, answers, costs,
              HITIds, usedWorkers, actions, agentActions,
              beliefs, ballots)  # GENERATES experiment.dump file

    #We keep the observations around for RL purposes
    #observations0 = [[] for i in range(0, numberOfProblems)]
    #observations1 = [[] for i in range(0, numberOfProblems)]

    #What is this for?
    observations = []
    for i in range(numberOfProblems):
        observations.append([])
        for j in range(numberOfWorkerPools):
            observations[i].append([])

    #gammas = ballots.calcAverageGammas() #Gammas contains number of worker pool averages

    print("Reading Policy")

    ###########################################################
    #Read the policy that we will begin with
    #You can choose a policy that's already been learned or learn
    #a new one
    ###########################################################
    #genPOMDP(filename,reward,costList,gammaList,numberofWorkerpools)


    fpipe = open('pipe.info','r')
    fpipe.readline()
    pathForPolicy = fpipe.readline().rstrip()
    fpipe.close()

    numdiffs = int((numStates - 1) / 2)

    policy = readPolicy(f"ModelLearning/Policies/W1_COST-500.policy", numStates)
    #policy = readPolicy(f"ModelLearning/Policies/unclassified.policy", numStates)

    '''
    try:
        policy = readPolicy(pathForPolicy + "out.policy", numStates)
    except:
        print("Generating POMDP")
        genPOMDP('log/pomdp/rl.pomdp', value * -1, costList, gammaList, numberOfWorkerPools)
        #Solve the POMDP

        zmdpDumpfile = open('log/pomdp/zmdpDump', 'w')
        subprocess.call('%s solve %s -o %s -t %d' % (
                ZMDPPATH,
                'log/pomdp/rl.pomdp',
                pathForPolicy + 'out.policy',
                timeLearning),
                        stdout=zmdpDumpfile,
                        shell=True,
                        executable="/bin/bash")
        zmdpDumpfile.close()

                        #Read the policy that we will begin with
        policy = readPolicy(pathForPolicy + 'out.policy', numStates)
        print(policy.keys())

    print("POMDP Generated, POMDP Solved, Policy read")
    print("Initialization Complete")
    '''

    fAns = open('SimulatedData/trueQuestionAnswers.txt', 'r')
    trueAnswers = [int(line.rstrip().split("\t")[1]) for line in fAns][0:numberOfProblems]
    fAns.close()

    unclassified_num = 0
    
    #Begin the experiment
    iterationNumber = -1
    while numberLeft(statuses) != 0:
        iterationNumber += 1
        # availableQuestions0 = []
        # availableQuestions1 = []
        availableQuestions = [[] for _ in range(numberOfWorkerPools)]
        if debug:
            print("Waiting 1 seconds")
        #time.sleep(1)
        print("Beginning iteration %d with %d questions remaining" % (iterationNumber, numberLeft(statuses)))
        #First we read from file all our data.
        (nop, statuses, answers, costs, HITIds, usedWorkers, actions, agentActions, beliefs, ballots) = unSerialize()
        #Loop through all the problems regardless of the ones left. Can maintain a set of problems left to speed up
        for i in range(0, numberOfProblems):
            if statuses[i] == DONE:
                continue
            elif statuses[i] == READY_FOR_ACTION:
                beliefState = beliefs[i]
                bestAction = findBestAction(actions, policy, beliefState)
                bestAction = int(bestAction)
                agentActions[i] = bestAction


                #Depending on the ballot0/ballot1, push to one or the other workflow
                if bestAction < numberOfWorkerPools:
                    availableQuestions[int(bestAction)].append(i)  # add task to collect vote

                    costs[i] += costList[bestAction]
                    statuses[i] = WAITING_FOR_TURKER

                elif bestAction == SUBMITZERO or bestAction == SUBMITONE:
                    if bestAction == SUBMITZERO:
                        answers[i] = 0
                    else:
                        answers[i] = 1

                    statuses[i] = DONE
            elif statuses[i] == WAITING_FOR_TURKER:
                continue

        #WHY BLOCKING THE CODE?
        for i in range(numberOfWorkerPools):
            while True:
                try:
                    mkdir('locks/aql' + str(i) + 'lock')
                    break
                except OSError:
                    pass

        #We only append to these files.
        aql = [open('log/aql'+str(i),'a') for i in range(numberOfWorkerPools)]

        for i in range(numberOfWorkerPools):
            for aq in availableQuestions[i]:
                aql[i].write('%d' % aq)
                HITIds.append(mt.createHIT(i,URL,nameOfTask,priceList[i],taskDuration,31536000,1,aq))

        #Close files and release locks
        for i in range(numberOfWorkerPools):
            aql[i].close()
            rmdir('locks/aql' + str(i) + 'lock')

        current_items_ids = []

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

                        current_items_ids.append(pn)

                        if observation in choices[pn]:
                            observation = choices[pn][observation]  # get vote class
                        else:
                            raise ValueError(observation)
                            print("Never go here!")
                            inversechoices[pn][choicesindex[pn]] = observation
                            choices[pn][observation] = choicesindex[pn]
                            choicesindex[pn] += 1
                            observation = choices[pn][observation]
                        
                        workerId = mt.getWorker(assignment)
                        mt.approveAssignment(assignment)


                        # save vote
                        items_votes[pn][workerId] = observation
                    #end for

                    #Now that we have an observation, we need to update our belief
                    #beliefs[pn] = updateBelief(beliefs[pn],
                    #                          observation, difficulties,
                    #                          ballots.getWorkerGammaGivenPool(workerId,
                    #                                                          agentActions[pn]))
                    # for 1 pool, pool # always gonna be 0



                    f = open('log/results/observations%dw%d' % (pn,agentActions[pn]),'a+')
                    observations[pn][agentActions[pn]].append((observation,workerId))
                    f.write('%s\t%s\n' % (inversechoices[pn][observation],workerId))
                    f.close()

                    mt.disposeHIT(HITId) 
                    statuses[pn] = READY_FOR_ACTION
                else:
                    nextHITIds.append(HITId)
        HITIds = nextHITIds

        estimated_error_rates = get_worker_error_rate_estimation(items_votes)

        # after votes collected, estimate error rates and update belief
        for item_id in current_items_ids:
            last_vote = list(items_votes[item_id].values())[-1]
            last_worker_id = list(items_votes[item_id])[-1]
            beliefs[item_id] = updateBelief(beliefs[item_id], last_vote, difficulties,
                                     get_worker_error_rate(last_worker_id, estimated_error_rates, avg_error_rate))



        #Write our world state to file in case something crashes and burns
        serialize(numberOfProblems, statuses, answers, 
                  costs, HITIds, usedWorkers,
                  actions, agentActions, beliefs, ballots)

    # end while

    fp = open('pipe.info','r')
    pathToLog = fp.readline().rstrip()
    path = fp.readline().rstrip()
    fp.close()

    '''
    f = open('log/results/answers', 'w') 
    for i in range(numberOfProblems):
        f.write('%s\n' % inversechoices[i][answers[i]])
    f.close()
    '''

    if not isLiveExperiment:
        #fDiff = open('SimulatedData/DifficultiesFileBell','r')
        fDiff = open('Experiments/Difficulties05','r')
        trueDifficulties = [float(line.rstrip().split(",")[1]) for line in fDiff][0:numberOfProblems]
        fDiff.close()

        #fAns = open('SimulatedData/trueQuestionAnswers.txt','r')
        #trueAnswers = [int(line.rstrip().split("\t")[1]) for line in fAns][0:numberOfProblems]
        #fAns.close()

        fSim = open('SimulatedData/WorkerPool.info','r')
        linesInSim = [line.rstrip() for line in fSim]
        totalWorkersOriginal = int(linesInSim[1]) + int(linesInSim[4])
        fSim.close()

        fGammas = open('SimulatedData/trueWorkerGammas.txt','r')
        trueGammas = [float(line.rstrip().split("\t")[1]) for line in fGammas][0:totalWorkersOriginal]
        fGammas.close()

        answers = list(map(lambda x: int(x),answers))
        accuracy = float((numberOfProblems - reduce(lambda x,y: x+y,map(lambda x,y: abs(x-y),answers,trueAnswers)))*100.0)/numberOfProblems

        f = open(pathToLog,'w')
        fc = open(path + "costs",'a')
        fa = open(path + "accuracies",'a')

        #f.write(f"Expert cost: {expert_cost}\n")
        #f.write(f"Unclassified Num: {unclassified_num}\n")
        f.write('Number of Worker Pools: %d\n'% numberOfWorkerPools)
        f.write('Number of Workers in Pool 1: %d\n' %int(linesInSim[1]))
        f.write('Original mean,standardDev for Pool 1: %f,%f\n'%(float(linesInSim[2]),float(linesInSim[3])))
        f.write('Number of Workers in Pool 2: %d\n'%int(linesInSim[4]))
        f.write('Original mean,standardDev for Pool 2: %f,%f\n'%(float(linesInSim[5]),float(linesInSim[6])))
        f.write('Number of Problems: %d\n'%numberOfProblems)
        f.write('ProblemNo,Answer,TrueAnswer,Cost,Difficulty,TrueDifficulty\n')
        for i in range(numberOfProblems):
            f.write('%d,%d,%d,%f,%f,%f\n' % (i,int(answers[i]),trueAnswers[i],costs[i],float(getMostLikelyDifficulty(beliefs[i], difficulties)),trueDifficulties[i]))
        f.write('WorkerID,Gamma,TrueGamma\n')
        for workerPool in ballots.workersToGammas:
            for workerId in workerPool:
                f.write('%d,%f,%f\n' % (int(workerId),workerPool[workerId],trueGammas[int(workerId)]))
        f.write('Average Cost: %f\n' % average(costs))
        f.write('Accuracy: %f\n' % accuracy)
        zipDiff = sorted(zip(trueDifficulties,map(lambda x,y: abs(x-y),answers,trueAnswers)))
        zipCost = sorted(zip(trueDifficulties,costs))
        for i in range(10):
            tempZipDiff = [term[1] for term in zipDiff[i*10:(i+1)*10]]
            tempZipCost = [term[1] for term in zipCost[i*10:(i+1)*10]]
            accuracyTerm = reduce(lambda x,y:x+y,tempZipDiff)
            partialAccuracy = float((10 - accuracyTerm)*100.0)/10
            f.write('Accuracy %d: %f\n' %(i,partialAccuracy))
            f.write('Cost %d: %f\n' %(i,average(tempZipCost)))
            if (i == 9):
                fc.write('%f\n' % average(tempZipCost))
                fa.write('%f\n' % partialAccuracy)
            else:
                fc.write('%f,' % average(tempZipCost))
                fa.write('%f,' % partialAccuracy)
        f.close()
        fc.close()
        fa.close()

    else:
        '''
        REAL DATA CODE
        fAns = open('client/groundTruths','r')
        fpp = open('questionsLookUp','r')
        mapping = {}
        imapping = {}
        for line in fpp:
            l = list(map(int,line.rstrip().split("\t")))
            mapping[l[0]] = l[1]
            imapping[l[1]] = l[0]
    
        trueAnswersTemp = [int(line.rstrip().split(" ")[1]) for line in fAns]#[0:numberOfProblems]
        trueAnswers = 150*[-1]
        for prob in range(numberOfProblems):
            trueAnswers[prob] = trueAnswersTemp[imapping[prob]]
        fAns.close()
        fpp.close()
        answers = list(map(lambda x: int(x),answers))
        accuracy = float((numberOfProblems - reduce(lambda x,y: x+y,map(lambda x,y: abs(x-y),answers,trueAnswers)))*100.0)/numberOfProblems
        f = open(pathToLog,'w')
        f.write('ProblemNo,Answer,TrueAnswer,Cost,Difficulty\n')
        for i in range(numberOfProblems):
            f.write('%d,%d,%d,%f,%f\n' % (i,int(answers[i]),trueAnswers[i],costs[i],float(getMostLikelyDifficulty(beliefs[i], difficulties))))
        f.write('WorkerID,Gamma\n')
        for workerPool in ballots.workersToGammas:
            for workerId in workerPool:
                f.write('%d,%f\n' % (int(workerId),workerPool[workerId]))
        f.write('Average Cost: %f\n' % average(costs))
        f.write('Accuracy: %f\n' % accuracy)
        f.close()
        '''

    return costs, answers
