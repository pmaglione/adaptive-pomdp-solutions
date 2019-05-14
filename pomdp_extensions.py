import random
from copy import deepcopy
import numpy as np
import algorithms_utils as alg_utils
import pandas as pd
from results_utils import *

import python_glad.glad_bin as glad_bin

# ACTIONS
CONST_NO_ANSWER = -1
CONST_REQUEST_VOTE = 0
CONST_SUBMIT_ZERO = 1
CONST_SUBMIT_ONE = 2

# policy and beliefs management
def dot(v1, v2):
    sum = 0.0
    for i in range(0, len(v1)):
        if v1[i] == '*' or v2[i] == '*':
            continue
        sum += v1[i] * v2[i]
    return sum

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

def normalize(array):
    sum = 0.0
    for i in range(0, len(array)):
        sum += array[i]  # when initialize all in 1, equivalent to 1 * numStates
    for i in range(0, len(array)):
        array[i] = array[i] / sum  # when initialize all in 1, equivalent to 1 / numStates
    return array

def calcAccuracy(gamma, d):
    return (1.0 / 2) * (1.0 + (1.0 - d) ** gamma)

def updateBelief(prevBelief, observation, difficulties, gamma):
    newBeliefs = []
    numDiffs = len(difficulties)
    for i in range(0, 2):
        for j in range(0, numDiffs):
            diff = difficulties[j]
            state = (i * numDiffs) + j

            if observation == i:
                newBeliefs.append(calcAccuracy(gamma, diff) * prevBelief[state])
            else:
                newBeliefs.append((1 - calcAccuracy(gamma, diff)) * prevBelief[state])

    newBeliefs.append(0.0)
    normalize(newBeliefs)
    return newBeliefs

def round_to_3(value):
    return round(value, 3)

def are_items_unresolved(answers):
    return any(answer == CONST_NO_ANSWER for answer in answers)


def get_unresolved_items(answers):
    return [key for key, answer in enumerate(answers) if answer == CONST_NO_ANSWER]


def get_random_worker(workers_error_rates, item_id, votes):
    item_votes = votes[item_id].copy()
    worker_ids_used = item_votes.keys()
    workers_ids_range = [k for k, v in enumerate(workers_error_rates)]
    workers_ids_unused = [val for val in workers_ids_range if val not in worker_ids_used]

    if (len(workers_ids_unused) == 0):
        used = len(worker_ids_used)
        ranges = len(workers_ids_range)
        unu = len(workers_ids_unused)
        print(f'used: {used}')
        print(f'workers: {ranges}')
        print(f'unused: {unu}')
        raise ValueError("Unused empty!?")

    selected_worker_id = np.random.choice(workers_ids_unused)
    error_rate = workers_error_rates[selected_worker_id]

    return selected_worker_id, error_rate


def get_accuracy(item_diff, worker_error_rate):
    return (1.0 / 2) * (1.0 + (1.0 - item_diff) ** worker_error_rate)


def get_worker_vote(item_id, items_votes, items_difficulties, items_gt, workers_error_rates):
    selected_worker_id, worker_error_rate = get_random_worker(workers_error_rates, item_id, items_votes)
    worker_acc = get_accuracy(items_difficulties[item_id], worker_error_rate)

    if np.random.binomial(1, worker_acc):
        return selected_worker_id, items_gt[item_id]
    else:
        return selected_worker_id, 1 - items_gt[item_id]


def get_error_rate(worker_id, estimated_error_rates, avg_error_rate):
    if worker_id in estimated_error_rates.keys():
        return estimated_error_rates[worker_id]
    elif len(estimated_error_rates) != 0:
        return sum(estimated_error_rates.values()) / len(estimated_error_rates)  # AVG over known workers
    else:
        return avg_error_rate


def get_worker_error_rate(worker_id, estimated_error_rates, avg_error_rate, estimate_after, have_submitted):
    if estimate_after:
        if have_submitted:
            return get_error_rate(worker_id, estimated_error_rates, avg_error_rate)
        else:
            return avg_error_rate
    else:
        return get_error_rate(worker_id, estimated_error_rates, avg_error_rate)


def get_workers_order(votes):
    worker_order = 0
    worker_to_order = {}
    order_to_worker = {}
    num_votes = 0
    num_workers = 0
    num_items = len(votes)

    for item_id, item_votes in votes.items():
        for worker_id, vote in item_votes.items():
            num_votes += 1
            if worker_id not in worker_to_order.keys():
                worker_to_order[worker_id] = worker_order
                order_to_worker[worker_order] = worker_id
                worker_order += 1

    num_workers = len(worker_to_order)

    return num_items, num_workers, num_votes, worker_to_order, order_to_worker


def writeToEMFormat(votes, path_em_input_file):
    num_items, num_workers, num_votes, worker_to_order, order_to_worker = get_workers_order(votes)

    outputfile = open(path_em_input_file, 'w')
    prior_beta = 0.5
    # file headers
    outputfile.write('%d %d %d %f\n' % (num_votes, num_workers, num_items, prior_beta))

    # votes
    for item_id, item_votes in votes.items():
        for worker_id, vote in item_votes.items():
            outputfile.write('%d %d %d\n' % (item_id, worker_to_order[worker_id], vote))

    outputfile.close()

    return worker_to_order, order_to_worker


def get_worker_error_rate_estimation(votes, path_em_input_file):
    # min 2 votes per item
    if all(len(v) >= 2 for k, v in votes.items()):
        worker_to_order, order_to_worker = writeToEMFormat(votes, path_em_input_file)

        error_rates = glad_bin.estimate(path_em_input_file)

        return {order_to_worker[worker_order]: abs(error_rate) for worker_order, error_rate in error_rates.items()}
    else:
        return {}

def get_error_rate(worker_id, estimated_error_rates, avg_error_rate):
    if worker_id in estimated_error_rates.keys():
        return estimated_error_rates[worker_id]
    elif len(estimated_error_rates) != 0:
        return sum(estimated_error_rates.values()) / len(estimated_error_rates)  # AVG over known workers
    else:
        return avg_error_rate

def get_worker_error_rate(worker_id, estimated_error_rates, avg_error_rate, estimate_after, have_submitted):
    if estimate_after:
        if have_submitted:
            return get_error_rate(worker_id, estimated_error_rates, avg_error_rate)
        else:
            return avg_error_rate
    else:
        return get_error_rate(worker_id, estimated_error_rates, avg_error_rate)

# Generate discrete difficulties between 0 and 1 at diffInterval steps
def getDifficulties(diffInterval):
    difficulties = []
    numDiffs = int(1.0 / diffInterval + 1)
    for i in range(0, numDiffs):
        difficulties.append(round(diffInterval * i, 1))
    return difficulties

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

# data utils
'''
    items_num - number of items
    possitive_percentage - [0,1] percentage of possitive items
'''


def generate_gold_data(items_num, possitive_percentage):
    pos_items_number = int(round(((possitive_percentage * 100) * items_num) / 100))
    gold_data = ([1] * pos_items_number) + ([0] * (items_num - pos_items_number))
    random.shuffle(gold_data)

    return gold_data


def solve(num_states, states_difficulties, avg_error_rate, policy, workers_error_rates, items_difficulties, items_gt,
          estimate_after=True, path_em_input_file='./log/em/ballots.eminput'):

    num_items = len(items_gt)

    actions = range(0, 3)  # 0,1,2
    # states_difficulties = getDifficulties(0.1)

    items_votes = {}
    for item_id in range(num_items):
        items_votes[item_id] = {}

    # init beliefs
    belief = [1 for i in range(num_states)]
    belief[num_states - 1] = 0  # last states = 0, terminating state
    belief = normalize(belief)
    beliefs = [deepcopy(belief) for i in range(num_items)]

    answers = [CONST_NO_ANSWER for i in range(0, num_items)]
    unresolved_items = get_unresolved_items(answers)

    iteration_number = 0
    while are_items_unresolved(answers):
        iteration_number += 1

        items_to_vote = []

        for item_id in unresolved_items:
            beliefState = beliefs[item_id]
            bestAction = findBestAction(actions, policy, beliefState)
            bestAction = int(bestAction)

            if bestAction == CONST_REQUEST_VOTE:
                items_to_vote.append(item_id)
            elif bestAction == CONST_SUBMIT_ZERO or bestAction == CONST_SUBMIT_ONE:
                if bestAction == CONST_SUBMIT_ZERO:
                    answers[item_id] = 0
                else:
                    answers[item_id] = 1
        # end for

        unresolved_items = get_unresolved_items(answers)
        unresolved_items_num = len(unresolved_items)
        have_submitted = unresolved_items_num != num_items

        for item_to_vote in items_to_vote:
            worker_id, vote = get_worker_vote(item_to_vote, items_votes, items_difficulties, items_gt,
                                              workers_error_rates)
            items_votes[item_to_vote][worker_id] = vote

        estimated_error_rates = get_worker_error_rate_estimation(items_votes, path_em_input_file)

        for item_id in items_to_vote:
            last_vote = list(items_votes[item_id].values())[-1]
            last_worker_id = list(items_votes[item_id])[-1]
            beliefs[item_id] = updateBelief(beliefs[item_id], last_vote, states_difficulties,
                                            get_worker_error_rate(last_worker_id, estimated_error_rates, avg_error_rate,
                                                                  estimate_after, have_submitted))

    # end while

    return answers, items_votes


# extensions

def generate_worker_error_rates(workers_num, dist_name, dist_mean, dist_std):
    if dist_name == "Normal":
        return np.random.normal(dist_mean, dist_std, workers_num)
    elif dist_name == "Beta":
        return np.random.beta(dist_mean, dist_std, workers_num)

def run_base_case(items_num, positive_percentage, item_difficulty, workers_num, avg_workers_error_rate,
                  dist_name, dist_mean, dist_std, states_num, policy_path, output_file, moment_error_estimations,
                  fncs, fpcs, get_policy_name_fn, columns_to_print, path_em_input_file='./log/em/ballots.eminput',
                  iterations_num=10):

    columns = ['name', 'num_workers', 'workers_distribution', 'policy_name', 'num_items', 'data_bal', 'items_diff',
               'num_states', 'cost', 'cost_std', 'recall', 'recall_std', 'precision', 'precision_std', 'loss',
               'loss_std', 'f1', 'f1_std', 'fbeta', 'fbeta_std', 'estimate_after', 'avg_error_rate',
               'wce', 'wce_std', 'fnc', 'fpc']

    items_difficulties = [item_difficulty] * items_num
    items_ground_truth = generate_gold_data(items_num, positive_percentage)
    workers_error_rates = generate_worker_error_rates(workers_num, dist_name, dist_mean, dist_std)

    state_diff = getDifficulties(0.1)

    header = True

    for moment_error_estimation in moment_error_estimations:
        for fnc in fncs:
            for fpc in fpcs:
                total_results = []

                policy_name = get_policy_name_fn(items_num, fnc, fpc)
                policy = readPolicy(policy_path + policy_name, states_num)

                losses = []
                recalls = []
                precisions = []
                costs = []
                f_ones = []
                f_betas = []
                wces = []

                for _ in range(iterations_num):
                    answers, items_votes = solve(states_num, state_diff, avg_workers_error_rate, policy,
                                                 workers_error_rates, items_difficulties, items_ground_truth,
                                                 moment_error_estimation, path_em_input_file)

                    costs.append(np.mean([len(v) for k, v in items_votes.items()]))

                    loss, recall, precision, f1, beta, f_beta, wce = alg_utils.Metrics.compute_metrics(answers,
                                                                                                   items_ground_truth,
                                                                                                   -1 * fnc, -1 * fpc)
                    losses.append(loss)
                    recalls.append(recall)
                    precisions.append(precision)
                    f_ones.append(f1)
                    f_betas.append(f_beta)
                    wces.append(wce)
                    # end for iterations

                result = [f"base-fnc{fnc}-fpc{fpc}", workers_num, dist_name + f"({dist_mean},{dist_std})",
                          policy_name, items_num, positive_percentage,
                          item_difficulty, states_num, round_to_3(np.mean(costs)), round_to_3(np.std(costs)),
                          round_to_3(np.mean(recalls)), round_to_3(np.std(recalls)),
                          round_to_3(np.mean(precisions)), round_to_3(np.std(precisions)), round_to_3(np.mean(losses)),
                          round_to_3(np.std(losses)),
                          round_to_3(np.mean(f_ones)), round_to_3(np.std(f_ones)), round_to_3(np.mean(f_betas)),
                          round_to_3(np.std(f_betas)), moment_error_estimation, avg_workers_error_rate,
                          round_to_3(np.mean(wces)), round_to_3(np.std(wces)), fnc, fpc]

                total_results.append(result)

                #  create or append
                df = pd.DataFrame(total_results, columns=columns)
                mode = 'w' if header else 'a'
                df.to_csv(output_file, mode=mode, index=False, header=header)
                header = False

        # end for fncs
    #end for moments

    data = pd.read_csv(output_file)
    elems_t = data[data.estimate_after == True]
    elems_f = data[data.estimate_after == False]

    elems = []
    labels = []
    for fnc in fncs:
        elems_t_filtered = elems_t[elems_t.fpc == fnc]
        elems_f_filtered = elems_f[elems_f.fpc == fnc]

        elems.append(elems_t_filtered)
        elems.append(elems_f_filtered)
        labels.append(f"After, FPN={fnc}")
        labels.append(f"Before, FPN={fnc}")

    plot_elems_lines(elems, fncs, columns_to_print, "FNC", labels)



def run_vary_num_states(items_num, positive_percentage, item_difficulty, workers_num, avg_workers_error_rate,
                  dist_name, dist_mean, dist_std, states_nums, states_diffs, policy_path, output_file,
                moment_error_estimations, fncs, fpcs, get_policy_name_fn, columns_to_print):

    columns = ['name', 'num_workers', 'workers_distribution', 'policy_name', 'num_items', 'data_bal', 'items_diff',
               'num_states', 'cost', 'cost_std', 'recall', 'recall_std', 'precision', 'precision_std', 'loss',
               'loss_std', 'f1', 'f1_std', 'fbeta', 'fbeta_std', 'estimate_after', 'avg_error_rate',
               'wce', 'wce_std', 'fnc', 'fpc']

    items_difficulties = [item_difficulty] * items_num
    items_ground_truth = generate_gold_data(items_num, positive_percentage)
    workers_error_rates = generate_worker_error_rates(workers_num, dist_name, dist_mean, dist_std)

    header = True

    for ind, states_num in enumerate(states_nums):
        for moment_error_estimation in moment_error_estimations:
            for fnc in fncs:
                for fpc in fpcs:
                    total_results = []

                    state_diff = states_diffs[ind]

                    policy_name = get_policy_name_fn(states_num, fnc, fpc)
                    policy = readPolicy(policy_path + policy_name, states_num)

                    losses = []
                    recalls = []
                    precisions = []
                    costs = []
                    f_ones = []
                    f_betas = []
                    wces = []

                    for _ in range(10):
                        answers, items_votes = solve(states_num, state_diff, avg_workers_error_rate, policy,
                                                     workers_error_rates, items_difficulties, items_ground_truth,
                                                     moment_error_estimation)

                        costs.append(np.mean([len(v) for k, v in items_votes.items()]))

                        loss, recall, precision, f1, beta, f_beta, wce = alg_utils.Metrics.compute_metrics(answers,
                                                                                                   items_ground_truth,
                                                                                                   -1 * fnc,
                                                                                                   -1 * fpc)
                        losses.append(loss)
                        recalls.append(recall)
                        precisions.append(precision)
                        f_ones.append(f1)
                        f_betas.append(f_beta)
                        wces.append(wce)
                        # end for iterations

                    result = [f"diff-states-fnc{fnc}-fpc{fpc}", workers_num, dist_name + f"({dist_mean},{dist_std})",
                              policy_name, items_num, positive_percentage,
                              item_difficulty, states_num, round_to_3(np.mean(costs)), round_to_3(np.std(costs)),
                              round_to_3(np.mean(recalls)), round_to_3(np.std(recalls)),
                              round_to_3(np.mean(precisions)), round_to_3(np.std(precisions)),
                              round_to_3(np.mean(losses)),
                              round_to_3(np.std(losses)),
                              round_to_3(np.mean(f_ones)), round_to_3(np.std(f_ones)), round_to_3(np.mean(f_betas)),
                              round_to_3(np.std(f_betas)), moment_error_estimation, avg_workers_error_rate,
                              round_to_3(np.mean(wces)), round_to_3(np.std(wces)), fnc, fpc]

                    total_results.append(result)

                    #  create or append
                    df = pd.DataFrame(total_results, columns=columns)
                    mode = 'w' if header else 'a'
                    df.to_csv(output_file, mode=mode, index=False, header=header)
                    header = False
    # end for

    data = pd.read_csv(output_file)
    #elems_base = data[data.name.str.startswith('base-')]
    total_elems = data[data.name.str.startswith('diff-states')]
    num_states = total_elems.num_states.unique()
    fncs = total_elems.fnc.unique()

    # num_states = np.append(num_states, 23)
    num_states.sort()
    num_states = [int(x) for x in num_states]

    for estimation_moment in [True, False]:
        labels = []
        values = []
        for num_st in num_states:
            for fnc in fncs:
                if estimation_moment:
                    label = f"After,#st={num_st},FPC={fnc}"
                else:
                    label = f"Before,#st={num_st},FPC={fnc}"

                labels.append(label)

                if num_st != 23:
                    elems_st = total_elems[total_elems.num_states == num_st]
                else:
                    #elems_st = elems_base
                    pass

                # compare results  on same
                elems_t = elems_st[elems_st.estimate_after == estimation_moment][elems_st.fpc == fnc]
                values.append(elems_t)

        # plot all
        print("\n")
        print(f"Estimation After?: {estimation_moment}")
        plot_elems_lines(values, fncs, columns_to_print, "FNC", labels)


def print_vars(total_elems):
    print(f"Num Workers: {total_elems.num_workers.unique()}")
    print(f"Workers Distribution: {total_elems.workers_distribution.unique()}")
    print(f"Workers Initial Error Rate: {total_elems.avg_error_rate.unique()}")
    print(f"Num Items: {total_elems.num_items.unique()}")
    print(f"Items Balance: {total_elems.data_bal.unique()}")
    print(f"Items Difficulty: {total_elems.items_diff.unique()}")
    print(f"Num states: {total_elems.num_states.unique()}")
    print(f"Error Estimations: {total_elems.estimate_after.unique()}")
    print(f"False Negative Costs: {total_elems.fnc.unique()}")
    print(f"False Positive Costs: {total_elems.fpc.unique()}")

def print_vars_base(total_elems):
    print(f"Num Workers: {total_elems.num_workers.unique()}")
    print(f"Workers Distribution: {total_elems.workers_distribution.unique()}")
    print(f"Workers Initial Error Rate: {total_elems.avg_error_rate.unique()}")
    print(f"Num Items: {total_elems.num_items.unique()}")
    print(f"Items Balance: {total_elems.data_bal.unique()}")
    print(f"Items Difficulty: {total_elems.items_diff.unique()}")
    print(f"Num states: {total_elems.num_states.unique()}")
    print(f"Error Estimations: {total_elems.estimate_after.unique()}")
    print(f"False Negative Costs == False Positive Costs: {total_elems.wrong_cost.unique()}")
