import numpy as np


def compute_ibs(train_events, train_times, test_events, test_times, surv_probs, bins):
    """
    Compute IBS with sksurv.metrics.integrated_brier_score.

    Args:
        train_events: (n_train,) bool, True means event happened.
        train_times: (n_train,) float
        test_events: (n_test,) bool
        test_times: (n_test,) float
        surv_probs: (n_test, n_times) survival probabilities at bin edges.
        bins: (n_times + 1,) bin edges.

    Returns:
        float: IBS (lower is better).
    """
    from sksurv.metrics import integrated_brier_score

    dt = np.dtype([("event", bool), ("time", float)])
    y_train = np.array(list(zip(train_events, train_times)), dtype=dt)
    y_test = np.array(list(zip(test_events, test_times)), dtype=dt)

    times = bins[1:].copy()

    # Clip to the common observable range of both sets so the metric is computable.
    t_min = max(np.min(train_times), np.min(test_times))
    t_max = min(np.max(train_times), np.max(test_times))
    mask = (times > t_min) & (times < t_max)
    if mask.sum() < 2:
        return float("nan")

    times = times[mask]
    surv_probs = surv_probs[:, mask]

    try:
        ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
    except Exception:
        ibs = float("nan")

    return ibs
