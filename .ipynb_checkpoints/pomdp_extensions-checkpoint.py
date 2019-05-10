from dai_pomdp.ModelLearning.utils import *
from dai_pomdp.Data import *
from dai_pomdp.Ballots import *
from dai_pomdp.helpers import *
import random
from copy import deepcopy
import numpy as np
import algorithms_utils as alg_utils
import pandas as pd
from results_utils import *

# ACTIONS
CONST_NO_ANSWER = -1
CONST_REQUEST_VOTE = 0
CONST_SUBMIT_ZERO = 1
CONST_SUBMIT_ONE = 2


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


def get_worker_error_rate_estimation(items_votes):
    # min 2 votes per item
    if all(len(v) >= 2 for k, v in items_votes.items()):
        workers_to_int = writeToEMFormat(items_votes)

        gammas, difficulties, posteriors = get_results()

        worker_keys = list(workers_to_int.keys())
        worker_int_keys = list(workers_to_int.values())

        return {worker_keys[worker_int_keys.index(key)]: gamma for key, gamma in enumerate(gammas)}
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
          estimate_after=True):
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

    iteration_number = 0
    while are_items_unresolved(answers):
        iteration_number += 1

        items_to_vote = []
        unresolved_items = get_unresolved_items(answers)
        unresolved_items_num = len(unresolved_items)

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

        have_submitted = unresolved_items_num != num_items

        for item_to_vote in items_to_vote:
            worker_id, vote = get_worker_vote(item_to_vote, items_votes, items_difficulties, items_gt,
                                              workers_error_rates)
            items_votes[item_to_vote][worker_id] = vote

        estimated_error_rates = get_worker_error_rate_estimation(items_votes)

        for item_id in items_to_vote:
            last_vote = list(items_votes[item_id].values())[-1]
            last_worker_id = list(items_votes[item_id])[-1]
            beliefs[item_id] = updateBelief(beliefs[item_id], last_vote, states_difficulties,
                                            get_worker_error_rate(last_worker_id, estimated_error_rates, avg_error_rate,
                                                                  estimate_after, have_submitted))

        # print(f"Num to vote: {len(items_to_vote)}")
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
                  fncs, fpcs, get_policy_name_fn):

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

                for _ in range(10):
                    answers, items_votes = solve(states_num, state_diff, avg_workers_error_rate, policy, workers_error_rates,
                                                 items_difficulties, items_ground_truth, moment_error_estimation)

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

            # end for
