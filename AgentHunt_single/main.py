from numpy import average
from MTurk.SimMT import SimMT
from MTurk.RealMT import RealMT
from solve import solve
##########################################
# Configuration
##########################################
#Number of Tasks
numberOfProblems = 100

#What's your task called?
nameOfTask = "Named Entity Recognition"

#Toggle debugging output
debug = True

#Do you want to relearn the POMDP after every task? If yes, set to False.
fastLearning = False

#How many seconds do you want to spend learning POMDPs if not fastLearning?
timeLearning = 300 

#How many seconds do you want to give workers to complete one task?
#For some reason, Amazon oddly starts the clock at 90 seconds, so if you want
#to give them 30 seconds, enter 120 seconds.
taskDuration = 150 

#value of a correct answer
VALUE = 100
#Payout per HIT
PRICE = 0.01

ZMDPPATH = '/Users/pmaglione/Repos/WorkerPoolSelection/ModelLearning/zmdp-1.1.7/bin/darwin18/zmdp'

# Must be absolute path to where EM is
EMPATH = '/Users/pmaglione/Repos/WorkerPoolSelection/EM/em'
#Address of your web server
URL = "http://<yourserverhere>/AgentHuntRelease1.0/client/client.php"

#AWS Access Key ID
AWSAKID = '<YOURAWSACCESSKEYIDHERE>'

#AWS Secret Access Key
AWSSAK = '<YOURAWSSECRETACCESSKEYHERE>'

#Sandbox
SANDBOX = False
############################################
#End Configuration
############################################

#IS THE ONLINE REINFORCEMENT LEARNING CONFIGURATION
numStates = 243 #(11 * 11 * 2 + 1 for terminal state)
mt = RealMT(numberOfProblems, AWSAKID, AWSSAK, sandbox = SANDBOX)
(costs, answers) = solve(mt,
                         numStates,
                         numberOfProblems, 
                         nameOfTask,
                         VALUE,
                         PRICE,
                         ZMDPPATH,
                         URL,
                         EMPATH,
                         fastLearning,
                         timeLearning,
                         taskDuration,
                         debug)

print "Average Cost:"
print average(costs)
