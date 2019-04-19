import numpy as np
import pandas as pd


# metrics
class Metrics:

    @staticmethod
    # k penalization for false negatives
    def compute_metrics(items_classification, gt, lr=1):
        # FP == False Inclusion
        # FN == False Exclusion
        fp = fn = tp = tn = 0.
        for i in range(len(gt)):
            gt_val = gt[i]
            cl_val = items_classification[i]

            if gt_val and not cl_val:
                fn += 1
            if not gt_val and cl_val:
                fp += 1
            if gt_val and cl_val:
                tp += 1
            if not gt_val and not cl_val:
                tn += 1

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        loss = (fp + (fn * lr)) / len(gt)
        f1 = (2 * precision * recall) / (precision + recall)
        beta = lr
        f_beta = (beta**2 + 1) * precision * recall / (beta**2 * precision + recall)

        return loss, recall, precision, f1, beta, f_beta


'''    
    Input: folder name of real world dataset, must have SQUARE file structure
    Output: folder, ground truth values,  workers accuracies
'''


def get_real_dataset_data(folder):
    path = f'data/{folder}/workerResponses.txt'
    votes = pd.read_csv(path, delimiter='\t', header=None)
    path = f'data/{folder}/groundTruth.txt'
    ground_truth = pd.read_csv(path, delimiter='\t', header=None)
    path = f'data/{folder}/categories.txt'
    categories = pd.read_csv(path, delimiter='\t', header=None)

    '''
        1. Get workers ids
        2. Iterate over worker responses and compare with ground truth
        3. Calculate worker accuracy 
    '''

    w_id_int = 0
    workers_accuracy = {}
    for worker_id in np.unique(votes[0]):

        worker_votes = votes[votes[0] == worker_id]
        worker_successes = 0
        worker_tasks = 0

        for index, worker_vote_info in worker_votes.iterrows():

            worker_vote_task = worker_vote_info[1]
            worker_vote = worker_vote_info[2]

            gt_val = ground_truth[ground_truth[0] == worker_vote_task][1]

            if len(gt_val) > 0:
                worker_tasks += 1
                if worker_vote == gt_val.item():
                    worker_successes += 1
        # endfor

        if worker_tasks > 0:
            w_acc = worker_successes / worker_tasks
            workers_accuracy[w_id_int] = [w_acc,
                                          w_acc]  # workaround to be equal as generated positive/negative accs structure
            w_id_int += 1

    classes_normalized = [x for x in categories[0]]
    gt_normalized = [classes_normalized.index(x) for x in ground_truth[1]]  # map categories to 0/1 by index

    return gt_normalized, workers_accuracy
