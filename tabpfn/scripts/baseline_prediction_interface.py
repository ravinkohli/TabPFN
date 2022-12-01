import tqdm
import numpy as np

from tabpfn.constants import DEFAULT_SEED

import traceback

def baseline_predict(metric_function, X_train, X_test, y_train, y_test, categorical_feats, metric_used=None, eval_pos=2, max_time=300, **kwargs):
    """
    Baseline prediction interface.
    :param metric_function:
    :param eval_xs:
    :param eval_ys:
    :param categorical_feats:
    :param metric_used:
    :param eval_pos:
    :param max_time: Scheduled maximum time
    :param kwargs:
    :return: list [np.array(metrics), np.array(outputs), best_configs] or [None, None, None] if failed
    """

    # eval_splits = list(zip(eval_xs.transpose(0, 1), eval_ys.transpose(0, 1)))
    # for eval_x, eval_y in tqdm.tqdm(eval_splits, desc='Calculating splits'+str(metric_function)+' '+str(eval_pos)):
    try:
        metric, output, best_config = metric_function(X_train, y_train, X_test, y_test,
                                                        categorical_feats,
                                                        metric_used=metric_used,
                                                        seed=kwargs.get("seed", DEFAULT_SEED),
                                                        max_time=max_time)
        return metric, output, best_config
    except Exception as e:
        print(f'There was an exception in {metric_function}')
        print(e)
        print(traceback.format_exc())
        return None, None, None