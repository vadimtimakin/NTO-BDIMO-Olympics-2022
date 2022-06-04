import random 
import numpy as np
import os


class SMAPE(object):

    def get_final_error(self, error, weight):
        return 1000 * (1 - error)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0

        for i in range(len(approx)):
            error_sum += abs(target[i] - approx[i]) / (target[i] + approx[i])

        return error_sum / len(approx), 1


class SMAPELoss(object):
    
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for i in range(len(targets)):
            d1 = 1000 * abs(targets[i] - approxes[i]) / (targets[i] + approxes[i]) 
            result.append((d1, -1))
        
        return result


def metric(true, pred):
    score = 1000 * (1 - sum([abs(t - p) / (t + p) for t, p in zip(true, pred)]) / len(pred))
    return score


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)