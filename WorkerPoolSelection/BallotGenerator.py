__author__ = 'shreya'

from random import random, gauss, shuffle, gammavariate,betavariate,normalvariate,uniform,seed
from ModelLearning.utils import generateBallot, generateBallotWithAcc
from time import time,sleep

import sys

if __name__ == "__main__":
    arg_numProblems = int(sys.argv[1])
    arg_balance = float(sys.argv[2])

    seed(42)

    poolInfo = open('SimulatedData/WorkerPool.info', 'r')

    # Vary these parameters during testing
    # In this same code uses different terminologies to same thing: problems, tasks, questions
    numOfQuestions = arg_numProblems
    numOfWorkerPools = int(poolInfo.readline())

    numOfWorkersInEachPool = []
    gammaMuPrior = [] #For Gamma-distributions, refers to 'Shape' parameter
    gammaSigmaPrior = [] #For Gamma distribution, refers to 'Scale' parameter
    difficulties = []

    #diffFile = open('Experiments/DifficultiesFileBell', 'r')
    diffFile = open('Experiments/Difficulties05', 'r')

    '''
    for i in range(10):
        for j in range(10):
            diff = uniform(float(i)/10,float((i+1))/10)
            difficulties.append(diff)
    
    shuffle(difficulties)
    
    for i in range(100):
            diffFile.write(str(i) + "," + str(difficulties[i]) + '\n')
    '''


    '''# For Difficulties sampled from a Bell-Curve
    for i in range(numOfQuestions):
        diff = betavariate(2,2) #Beta distribution with alpha = 2 and beta = 2
        difficulties.append(diff)
        diffFile.write(str(i) + "," + str(difficulties[i]) + '\n')
    
    
    
    # For Difficulties sampled from a Bi-modal Distribution
    for i in range(numOfQuestions):
        diff = betavariate(0.5,0.5) #Beta distribution with alpha = 2 and beta = 2
        difficulties.append(diff)
        diffFile.write(str(i) + "," + str(difficulties[i]) + '\n')
    
    
    diffFile.close()
    sleep(10)
    '''

    difficulties = [float(line.rstrip().split(",")[1]) for line in diffFile]

    for i in range(numOfWorkerPools):
        numOfWorkersInEachPool.append(int(poolInfo.readline()))
        gammaMuPrior.append(float(poolInfo.readline())) # When worker gammas are sampled from a Gamma distribution, this refers to 'Shape'
        gammaSigmaPrior.append(float(poolInfo.readline())) # When worker gammas are sampled from a Gamma distribution, this refers to 'Scale'

    workerGammas = open('SimulatedData/trueWorkerGammas.txt', 'w')
    trueAnswers = open('SimulatedData/trueQuestionAnswers.txt', 'w')

    #Counter for worker IDs
    wID = -1

    GammaSet = {}

    for wpool in range(numOfWorkerPools):
        for w in range(numOfWorkersInEachPool[wpool]):
            gamma = gammavariate(gammaMuPrior[wpool],gammaSigmaPrior[wpool])
            wID += 1
            workerGammas.write(str(wID) + '\t' + str(gamma) + '\n')
            GammaSet[wID] = gamma


    # print GammaSet
    seed(time())
    wID = -1

    for q in range(numOfQuestions):
        randNumForTrueAns = uniform(0,1)
        v = -1

        if randNumForTrueAns < arg_balance:
            v = 0
        else:
            v = 1

        trueAnswers.write(str(q) + '\t' + str(v) + '\n')

        for wpool in range(numOfWorkerPools):
            f = open('SimulatedData/w%dq%d' % (wpool, q), 'w')
            AnswersForPoolQuestion = []
            for w in range(numOfWorkersInEachPool[wpool]):
                wID += 1
                # print(str(wID) + '\n')
                gammaTemp = GammaSet[wID]
                #ADAPTED TO NORMAL DISTRIBUTION, dont affect formula
                workerAnswer = generateBallot(gammaTemp,difficulties[q],v)
                #workerAnswer = generateBallotWithAcc(gammaTemp, v)
                temp = workerAnswer,wID
                AnswersForPoolQuestion.append(temp)
            shuffle(AnswersForPoolQuestion)
            f.write(str(AnswersForPoolQuestion[0][0]) + '\t' + str(AnswersForPoolQuestion[0][1]))
            for i in range(1,numOfWorkersInEachPool[wpool]):
                f.write('\n' + str(AnswersForPoolQuestion[i][0]) + '\t' + str(AnswersForPoolQuestion[i][1]))
            f.close()
        wID = -1

    workerGammas.close()
    trueAnswers.close()