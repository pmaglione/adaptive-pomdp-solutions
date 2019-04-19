import os
import subprocess


def writeToEMFormat(items_votes):
    os.chdir("/Users/pmaglione/Repos/adaptive-pomdp-solutions/dai_pomdp")
    num_votes = sum([len(v) for k, v in items_votes.items()])
    worker_ids = []
    _ = [[worker_ids.append(w_id) for w_id in v.keys() if w_id not in worker_ids] for k, v in items_votes.items()]
    num_workers = len(worker_ids)
    num_items = len(items_votes)

    workers_to_int = {worker_id: key for key, worker_id in enumerate(list(set(worker_ids)))}

    outputfile = open('log/test/votes.eminput', 'w')
    outputfile.write('%d %d %d %d %d %f\n' % (num_votes, num_workers, num_items, 1, 1, 0.5))

    for item_id, item_votes in items_votes.items():
        for worker_id, vote in item_votes.items():
            outputfile.write('%d %d %d %d %d %f\n' % (item_id, workers_to_int[worker_id], 0, 0, vote, 0.5))

    outputfile.close()

    return workers_to_int


def get_results():
    EMPATH = '/Users/pmaglione/Repos/adaptive-pomdp-solutions/dai_pomdp/EM/em'
    os.chdir("/Users/pmaglione/Repos/adaptive-pomdp-solutions/dai_pomdp/EM")

    EMDone = False
    while not EMDone:
        outputfile = open('../log/test/results', 'w')
        process = subprocess.call(f"{EMPATH} ../log/test/votes.eminput", stdout=outputfile, shell=True)
        outputfile.close()
        inputfile = open('../log/test/results', 'r').read().split("\n")
        inputfile = inputfile[0:len(inputfile) - 1]
        for nextLine in inputfile:
            if 'Beta' in nextLine:
                EMDone = True

    gammas = []
    difficulties = []
    posteriors = []

    inputfile = open('../log/test/results', 'r').read().split("\n")
    inputfile = inputfile[0:len(inputfile) - 1]
    for i in range(0, len(inputfile)):
        nextLine = inputfile[i]
        if 'Iteration' in nextLine:
            continue
        # These are problem difficulties
        elif 'Beta' in nextLine:
            d = float(nextLine.split("=")[1].strip())
            difficulties.append(d)
        elif 'P' in nextLine:
            p = float(nextLine.split("=")[2].strip())
            posteriors.append(p)
        # These are worker gammas.
        else:
            g = float(nextLine)
            gammas.append(g)
    # end for

    return gammas, difficulties, posteriors


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


def get_worker_error_rate(worker_id, estimated_error_rates, avg_error_rate):
    if worker_id in estimated_error_rates.keys():
        return estimated_error_rates[worker_id]
    elif len(estimated_error_rates) != 0:
        return sum(estimated_error_rates.values()) / len(estimated_error_rates)  # AVG over known workers
    else:
        return avg_error_rate
