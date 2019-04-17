#from MT import MT
from MTurk.MT import MT
import numpy as np

class FakeHIT():
    def __init__(self):
        self.HITStatus = 'Reviewable'
    
###############################################################
# This class allows the dynamic polling of your own files as a substitute
# for Amazon Mechanical Turk
#
# Files should be named in the form w<workflowNumber>q<problemNumber> 
# Each lines in the file should be of the form
# <observation>\t<workerId>
# Files need to be named in the form w<workerPoolNumber>q<problemNumber>
# Observations are formatted in the same way as above
################################################################
class SimMT(MT):
    def __init__(self, numberOfProblems, numberOfPools, offset):
        self.usedWorkers = [[] for _ in np.arange(numberOfProblems)]
        #self.answers0 = [[] for i in np.arange(0, numberOfProblems)]
        #self.answers1 = [[] for i in np.arange(0, numberOfProblems)]
        self.answers = []
        for i in np.arange(numberOfProblems):
            self.answers.append([])
            for j in np.arange(offset,numberOfPools+offset):
                self.answers[i].append([])
                f = open("SimulatedData/w%dq%d" % (j,i),'r').read().split("\n")
                for entry in f:
                    entry = entry.split("\t")
                    vote = entry[0]
                    worker = entry[1]
                    self.answers[i][j-offset].append((vote,worker,1,i))


    def createHIT(self, poolNumber, question, title, reward, duration, lifetime,
                  max_assignments, problemNumber):
            return 'W%dX%d' % (poolNumber, problemNumber)

    def getHIT(self, HITId):
        return [FakeHIT()]

    def getAssignments(self, HITId, page_size, page_number):
        assignments = []
        HITId = HITId.split("X")
        poolNumber = int(HITId[0].split("W")[1])
        problemNumber = int(HITId[1])

        while True: # stops when can assign
                if len(self.answers[problemNumber][poolNumber]) > 1:
                    nextAssignment = self.answers[problemNumber][poolNumber].pop(0)
                else:
                    print("WARNING: ONE VOTE LEFT FOR PROB %d" % problemNumber)
                    nextAssignment = self.answers[problemNumber][poolNumber][0]
                if self.getWorker(nextAssignment) in self.usedWorkers[problemNumber] and len(self.answers[problemNumber][poolNumber]) > 1:
                    continue  # if worker already vote on task and remaining workers > 1
                else:
                    assignments.append(nextAssignment)
                    break
        return assignments

    def getObservation(self, assignment):
        (obs, worker, assignmentId, pn) = assignment
        return obs

    def getWorker(self, assignment):
        (obs, worker, assignmentId, pn) = assignment
        return worker

    def getProblemNumber(self, assignment):
        (obs, worker, assignmentId, pn) = assignment
        return pn

    def approveAssignment(self, assignment):
        return True

    def disposeHIT(self, HITId):
        return True

