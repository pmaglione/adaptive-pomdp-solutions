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

        return loss, recall, precision