import subprocess
import os
from numpy import average

class Ballots:

    def __init__(self, emp):
        self.EMPATH = emp
        self.ballots0 = []
        self.numQuestions = 0
        self.numWorkers = 0
        self.numLabels = 0
        self.workersToIntegers = {}
        self.integersToWorkers = {}
        self.workersToGammas0 = {}
        self.workersToNumberCompleted = {}

    def writeToEMFormat(self):        
        outputfile = open('log/em/ballots.eminput', 'w')
        outputfile.write('%d %d %d %d %f\n' % (self.numLabels, self.numWorkers, 
                                                self.numQuestions, 1, 0.5))
        
        for (problem, i) in zip(self.ballots0, 
                                range(0, self.numQuestions)):
            for (vote, workerId) in problem:
                outputfile.write('%d %d %d %d\n' % (
                        i, self.workersToIntegers[workerId], 0, vote))

        outputfile.close()

    def runEM(self):
        self.writeToEMFormat()

        EMDone = False
        while not EMDone:
            outputfile = open('log/em/emresults', 'w')
            subprocess.call('%s log/em/ballots.eminput' % self.EMPATH,
                            stdout=outputfile,
                            shell=True)
            outputfile.close()
            inputfile = open('log/em/emresults', 'r').read().split("\n")
            inputfile = inputfile[0:len(inputfile) - 1]
            for nextLine in inputfile:
                if 'Beta' in nextLine:
                    EMDone = True


        #Now we want to format the output to be nice

        gammaoutputs0 = open('log/em/gammas.emresults0', 'w')
        diffoutputs0 = open('log/em/diffs.emresults0', 'w')
        posterioroutputs = open('log/em/posteriors.emresults','w')

        inputfile = open('log/em/emresults', 'r').read().split("\n")
        inputfile = inputfile[0:len(inputfile) - 1]
        problemCount = 0
        workerCount = 0
        for i in range(0, len(inputfile)):
            nextLine = inputfile[i]
            if 'Iteration' in nextLine:
                continue
            elif 'Beta' in nextLine: #These are problem difficulties
                d = float(nextLine.split("=")[1].strip())

                diffoutputs0.write('%f\n' % d)
                problemCount += 1 #These are posteriors
            elif 'P' in nextLine:
                p = float(nextLine.split("=")[2].strip())
                posterioroutputs.write('%f\n' % p)
            else: #These are worker gammas. 
                g = float(nextLine)
                gammaoutputs0.write('%f\n' % g)
                workerCount += 1

        gammaoutputs0.close()
        diffoutputs0.close()
        posterioroutputs.close()
        
    def addQuestionAndRelearn(self, obs0, answer, diffs, fast = True):
        self.numQuestions += 1
        self.ballots0.append(obs0)

        for (vote, workerId) in obs0:
            if workerId not in self.workersToIntegers:
                self.workersToIntegers[workerId] = self.numWorkers
                self.integersToWorkers[self.numWorkers] = workerId
                self.numWorkers += 1
        self.numLabels += len(obs0)

        if not fast:
            self.runEM()
            self.readGammaResults()
        else:
            (diff0, diff1) = diffs
            for (vote, workerId) in obs0:
                if workerId not in self.workersToNumberCompleted:
                    self.workersToNumberCompleted[workerId] = 0
                if workerId not in self.workersToGammas0:
                    self.workersToGammas0[workerId] = 1.0
                alpha = 0.5 / (self.workersToNumberCompleted[workerId] + 1)    
                if answer == vote:
                    self.workersToGammas0[workerId] -= (diff0 * alpha)
                    if self.workersToGammas0[workerId] < 0.01:
                        self.workersToGammas0[workerId] = 0.01
                else:                    
                    self.workersToGammas0[workerId] += (1.0-diff0) * alpha
                self.workersToNumberCompleted[workerId] += 1


    def readGammaResults(self):
        infile0 = open('log/em/gammas.emresults0', 'r')
        infile0 = infile0.read().split("\n")
        infile0 = infile0[0:len(infile0) - 1]
        for i in range(0, len(infile0)):
            self.workersToGammas0[self.integersToWorkers[i]] = float(infile0[i])

    def getWorkerGamma0(self, workerId):
        if workerId in self.workersToGammas0:
            return self.workersToGammas0[workerId]
        else:
            (gamma0, gamma1) = self.calcAverageGammas()
            return gamma0

    def calcAverageGammas(self):

        gammas0 = [value for key,value in self.workersToGammas0.items()]

        if len(gammas0) == 0:
            avg0 = 1.0
        else:
            avg0 = average(gammas0)

        return avg0
