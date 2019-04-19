from numpy import average
from MTurk.SimMT import SimMT
#from MTurk.RealMT import RealMT
from solve import solve
import sys

if __name__ == "__main__":
    ##########################################
    # Configuration
    ##########################################

    arg_timeLearning = int(sys.argv[1]) #original paper set 300 seconds
    arg_numProblems = int(sys.argv[2])
    arg_isLiveInt = int(sys.argv[3])
    arg_isLiveExperiment = bool(arg_isLiveInt)
    arg_numStates = int(sys.argv[4])
    arg_expertCost = int(sys.argv[5])

    #Number of Tasks
    numberOfProblems = arg_numProblems

    #What's your task called?
    nameOfTask = "Named Entity Recognition"

    #Toggle debugging output
    debug = True

    #Do you want to relearn the POMDP after every task? If yes, set to False.
    fastLearning = False

    #How many seconds do you want to spend learning POMDPs if not fastLearning?
    timeLearning = arg_timeLearning

    #How many seconds do you want to give workers to complete one task?
    #For some reason, Amazon oddly starts the clock at 90 seconds, so if you want
    #to give them 30 seconds, enter 120 seconds.
    taskDuration = 150


    f = open('simulation.info','r')
    #How many worker pools would you like to have available?
    #Typical value is 2, where the worker pools correspond to unstarred, and starred.
    workerPools = int(f.readline().rstrip())

    #value of a incorrect answer
    VALUE = int(f.readline().rstrip())
    #Payout per HIT, for 1each worker pool.
    PRICE = [float(x) for x in f.readline().rstrip().split(",")]
    #Prior on Gamma Averages for each worker pool.
    GAMMA = [float(x) for x in f.readline().rstrip().split(",")]

    SCALEFACTOR = int(f.readline().rstrip())

    OFFSET = int(f.readline().rstrip())

    f.close()

    #Must be an absolute path to where ZMDP is
    #ZMDPPATH = '/Users/shreyarajpal/Desktop/MausamOld/AgentHuntUpdatedReleaseLiveExperiments/ModelLearning/zmdp-1.1.7/bin/darwin14/zmdp'
    #ZMDPPATH = '/Users/pmaglione/zmdp-1.1.7/bin/darwin18/zmdp'
    ZMDPPATH = '/Users/pmaglione/Repos/adaptive-pomdp-solutions/WorkerPoolSelection/ModelLearning/zmdp-1.1.7/bin/darwin18/zmdp'

    #Must be absolute path to where EM is
    #EMPATH = '/Users/shreyarajpal/Desktop/MausamOld/AgentHuntUpdatedReleaseLiveExperiments/EM/em'
    EMPATH = '/Users/pmaglione/Repos/adaptive-pomdp-solutions/WorkerPoolSelection/EM/em'

    #Address of your web server
    URL = "http://<yourserverhere>/AgentHuntRelease1.0/client/client.php"

    #AWS Access Key ID
    AWSAKID = '<YOURAWSACCESSKEYIDHERE>'

    #AWS Secret Access Key
    AWSSAK = '<YOURAWSSECRETACCESSKEYHERE>'

    #Sandbox
    SANDBOX = True

    #Simulation
    #SIMULATION = True
    ############################################
    #End Configuration
    ############################################

    numStates = arg_numStates #(11 * 11 * 2 + 1 for terminal state)
    if arg_isLiveExperiment:
        #mt = RealMT(numberOfProblems, AWSAKID, AWSSAK, sandbox = SANDBOX)
        mt = ""
    else:
        mt = SimMT(numberOfProblems,workerPools,OFFSET)

    (costs, answers) = solve(mt,
                             numStates,
                             numberOfProblems,
                             workerPools,
                             nameOfTask,
                             VALUE,
                             PRICE,
                             GAMMA,
                             SCALEFACTOR,
                             ZMDPPATH,
                             URL,
                             EMPATH,
                             fastLearning,
                             timeLearning,
                             taskDuration,
                             debug,
                             arg_isLiveExperiment,
                             arg_expertCost)

    print("Average Cost:")
    print(average(costs))
    print("Answers:")
    print(answers)
    for problemNumber in range(numberOfProblems):
        print("The answer to problem %d is %d and it cost %f." % (problemNumber,answers[problemNumber],costs[problemNumber]))
