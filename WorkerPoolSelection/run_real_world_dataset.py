from os import system
from algorithms_utils import *

gammas_dist = [(4.,0.42),(4.,0.15),(4.,0.4),(4.,0.14),(4.,0.5)]
datasets = ['BarzanMozafari', 'RTE', 'SpamCF', 'TEMP', 'WVSCM']

for key, dataset in enumerate(datasets):
    ground_truth, workers_accuracy = get_real_dataset_data(dataset)
    items_num = len(ground_truth)
    data_bal = sum(ground_truth) / items_num
    worker_num = len(workers_accuracy)

    shape, scale = gammas_dist[key]

    system(f"echo '{worker_num},{shape},{scale}|{worker_num},{shape},{scale}' > /Users/pmaglione/Repos/adaptive-pomdp-solutions/WorkerPoolSelection/Experiments/WorkerDistributionPairs")

    system(f"/bin/bash run_experiment.sh 300 {items_num} {data_bal}")

    #system(f"cp -R /Users/pmaglione/Repos/adaptive-pomdp-solutions/WorkerPoolSelection/Experiments/{worker_num},{shape},{scale},{worker_num},{shape},{scale}/500 /Users/pmaglione/Repos/adaptive-pomdp-solutions/WorkerPoolSelection/ResultsRealWorld/{dataset}")

    #system("rm -rf /Users/pmaglione/Repos/adaptive-pomdp-solutions/WorkerPoolSelection/Experiments/1000,4.00,0.30,1000,4.00,0.30")




