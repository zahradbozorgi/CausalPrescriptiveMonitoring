from typing import List
from statistics import mean
from math import sqrt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from copy import deepcopy

from models.base import BaseGenModel
from causal_estimators.base import BaseEstimator, BaseIteEstimator

STACK_AXIS = 0
CONF = 0.95
EPSILON = 1e-10
REGRESSION_SCORES = ['max_error', 'neg_mean_absolute_error', 'neg_median_absolute_error',
                     'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
REGRESSION_SCORE_DEF = 'r2'
CLASSIFICATION_SCORES = ['accuracy', 'balanced_accuracy', 'average_precision',
                         'f1',
                         'precision',
                         'recall', 'roc_auc']
CLASSIFICATION_SCORE_DEF = 'accuracy'

##### TIME CONFOUNDS ######
# elim_16 = [2,3,4,9]
# elim_17 = [2,159,160,161]
# elim_19 = [0,413,414,415,416,417]
# elim_20 = [1,21,22,23,24,25]

##### CASE CONFOUNDS ######
elim_161 = [0,1,10,11,12,13,14,15]
elim_171 = [0,1,154, 155, 156, 157, 158, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,174]
elim_191 = [418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436,
           437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
           456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
           475, 476, 477, 478, 479, 480, 481]
elim_201 = [0]

##### RESOURCE CONFOUNDS ######
elim_162 = []
elim_172 = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
           39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
           64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
           89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
           111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
           131, 132, 133, 134, 135, 136, 137, 138]
elim_192 = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
           84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
           107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
           127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
           147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
           167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
           187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
           207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
           227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
           247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
           267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
           287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306,
           307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
           327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346,
           347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
           367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386,
           387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406,
           407, 408, 409, 410]
elim_202 = [14, 15, 16, 17, 18, 19, 20]

elim_17 = elim_171+elim_172
elim_19 = elim_191+elim_192

##### EVENT ATTRIBUTE CONFOUNDS ######


def run_model_cv(gen_model: BaseGenModel, model: sklearn.base.BaseEstimator, model_name: str,
                 param_grid, n_seeds: int, model_type: str, n_folds=5,
                 scoring=None, rank_score=None, best_params=False, best_model=False,
                 ret_time=False):
    model_type = model_type.lower()
    if model_type == 'outcome' or model_type == 'direct':
        if scoring is None:
            scoring = REGRESSION_SCORES
        if rank_score is None:
            rank_score = REGRESSION_SCORE_DEF
    elif model_type == 'prop_score' or model_type == 'ps':
        if scoring is None:
            scoring = CLASSIFICATION_SCORES
        if rank_score is None:
            rank_score = CLASSIFICATION_SCORE_DEF
    else:
        raise ValueError('Invalid model_type: {}'.format(model_type))
    cv = GridSearchCV(model, param_grid, scoring=scoring, cv=n_folds, refit=rank_score)
    dfs = []
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed)
        # w = w[:100,:]
        # t = t[:100,:]
        # y = y[:100,:]
        if model_type == 'outcome':
            if t.ndim == 1:
                t = t[:, np.newaxis]
            elif t.ndim == 2:
                pass
            else:
                raise ValueError('Invalid dimension of t: {}'.format(t.ndim))
            y = y.squeeze()
            X = np.hstack((w, t))
            cv.fit(X, y)
        else:   # propensity score model_type
            t = t.squeeze()
            cv.fit(w, t)
        d = {k: v for k, v in cv.cv_results_.items() if 'split' not in k}
        dfs.append(pd.DataFrame(d))

    df = pd.concat(dfs, axis=0)
    params = df.params
    cols = [col for col, is_param in zip(df.columns, df.columns.str.startswith('param_')) if is_param]

    # mean over seeds
    if len(cols) > 0:
        agg_df = df.drop('params', axis='columns').groupby(cols).mean().reset_index()
    else:
        agg_df = pd.DataFrame(df.drop('params', axis='columns').mean()).transpose()
    agg_df.insert(0, model_type + '_model', model_name)
    agg_df.insert(1, 'params_' + model_type + '_model', pd.Series(params.iloc[:len(agg_df)], index=agg_df.index))

    # remove timing columns
    if not ret_time:
        time_cols = [col for col, is_time in zip(df.columns, df.columns.str.endswith('time')) if is_time]
        agg_df.drop(time_cols, axis='columns', inplace=True)

    if best_params or best_model:
        results = {'df': agg_df}
        rank_col = 'rank_test_{}'.format(rank_score)
        best_cv = agg_df[agg_df[rank_col] == agg_df[rank_col].min()]
        best_cv_params = best_cv.params.iloc[0]
        best_cv_model = model.set_params(**best_cv_params)
        if best_params:
            results['best_params'] = best_cv_model.get_params()
        if best_model:
            results['best_model'] = best_cv_model
        return results
    else:
        return agg_df


def calculate_outcome_model_scores(gen_model: BaseGenModel, estimator: sklearn.base.BaseEstimator,
                                   n_seeds: int, n_folds=5, ret='mean'):

    def result_key(score):
        return '{}fold_{}'.format(n_folds, score)

    results = {result_key(score): np.zeros(n_seeds) for score in REGRESSION_SCORES}
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed)
        # w = w[:100,:]
        # t = t[:100,:]
        # y = y[:100,:]
        if t.ndim == 1:
            t = t[:, np.newaxis]
        elif t.ndim == 2:
            pass
        else:
            raise ValueError('Invalid dimension of t {}'.format(t.ndim))
        X = np.hstack((w, t))
        cv_results = cross_validate(estimator, X, y, cv=n_folds, scoring=scores)
        for score in REGRESSION_SCORES:
            results[result_key(score)][seed] = cv_results['test_' + score].mean()

    ret = ret.lower()
    if ret == 'mean':
        mean_results = {'mean_{}'.format(k): v.mean() for k, v in results.items()}
        return mean_results
    elif ret == 'all' or 'seeds' in ret:
        return results


def calculate_metrics(gen_model: BaseGenModel, estimator: BaseEstimator,
                      n_seeds: int, conf_ints=True, return_ite_vectors=False,
                      ate=None, ite=None, goal="continuous", meta_est_name='', dataname='', confound_col=''):
    if ate is None:
        ate = gen_model.ate()
    # else:
    #     print('Already computed ate')
    if ite is None:
        ite = gen_model.ite().squeeze()
    # else:
    #     print('Already computed ite')
    fitted_estimators = []
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed, overlap=1.0)
        # w = w[:,:25]
        # t = t[:,:]
        # y = y[:,:]

        # remove one set of features
        if dataname == "bpic17":
            w = np.delete(w, elim_17, axis=1 - STACK_AXIS)
        if dataname == "bpic19":
            w = np.delete(w, elim_19, axis=1 - STACK_AXIS)
        # if dataname == 'bpic16':
        #     w = np.delete(w, elim_16, axis=1 - STACK_AXIS)
        # if dataname == 'bpic20':
        #     w = np.delete(w, elim_20, axis=1 - STACK_AXIS)

        # only train with one set of features
        # if dataname == "bpic17":
        #     w = w[:,elim_17]
        # if dataname == "bpic19":
        #     w = w[:,elim_19]
        # if dataname == 'bpic16':
        #     w = w[:,elim_16]
        # if dataname == 'bpic20':
        #     w = w[:,elim_20]

        estimator.fit(w, t, y)
        print('fitting done!')
        fitted_estimators.append(estimator.copy())

    w_test, t_test, (y0, y1) = gen_model.sample(seed=n_seeds+1, ret_counterfactuals=True, overlap=1.0)
    if dataname == "bpic17":
        w_test = np.delete(w_test, elim_17, axis=1 - STACK_AXIS)
    if dataname == "bpic19":
        w_test = np.delete(w_test, elim_19, axis=1 - STACK_AXIS)

    # if dataname == "bpic17":
    #     w_test = w_test[:, elim_17]
    # if dataname == "bpic19":
    #     w_test = w_test[:, elim_19]
    # if dataname == 'bpic16':
    #     w_test = w_test[:, elim_16]
    # if dataname == 'bpic20':
    #     w_test = w_test[:, elim_20]

    ate_metrics = calculate_ate_metrics(w_test, ate, fitted_estimators, conf_ints=conf_ints)
    print('ate metrics done!')

    is_ite_estimator = isinstance(estimator, BaseIteEstimator)
    if is_ite_estimator:
        ite_metrics = calculate_ite_metrics(w_test,t_test,y0,y1, ite, fitted_estimators, goal=goal,
                                            meta_est_name=meta_est_name, dataname=dataname, confound_col=confound_col)
        ite_mean_metrics = {'mean_' + k: np.mean(v) for k, v in ite_metrics.items()}
        ite_std_metrics = {'std_of_' + k: np.std(v) for k, v in ite_metrics.items()}

    metrics = ate_metrics
    if is_ite_estimator:
        metrics.update(ite_mean_metrics)
        metrics.update(ite_std_metrics)
        if return_ite_vectors:
            metrics.update(ite_metrics)
    return metrics


def calculate_ate_metrics(w_test: np.ndarray, ate: float, fitted_estimators: List[BaseEstimator], conf_ints=True):
    ate_estimates = [fitted_estimator.estimate_ate(w_test) for
                     fitted_estimator in fitted_estimators]
    mean_ate_estimate = mean(ate_estimates)
    ate_mean_abs_error = mean(abs(ate_estimate - ate) for ate_estimate in ate_estimates)
    print('ate:', ate)
    print('mean ate estimate', mean_ate_estimate)
    ate_bias = mean_ate_estimate - ate
    ate_abs_bias = abs(ate_bias)
    ate_squared_bias = ate_bias**2
    ate_variance = calc_variance(ate_estimates, mean_ate_estimate)
    ate_std_error = sqrt(ate_variance)
    ate_mse = calc_mse(ate_estimates, ate)
    ate_rmse = sqrt(ate_mse)

    metrics = {
        'ate_mean_abs_error': ate_mean_abs_error,
        'ate_bias': ate_bias,
        'ate_abs_bias': ate_abs_bias,
        'ate_squared_bias': ate_squared_bias,
        'ate_variance': ate_variance,
        'ate_std_error': ate_std_error,
        'ate_mse': ate_mse,
        'ate_rmse': ate_rmse,
    }

    if conf_ints:
        ate_conf_ints = [fitted_estimator.ate_conf_int(CONF) for
                         fitted_estimator in fitted_estimators]
        metrics['ate_coverage'] = calc_coverage(ate_conf_ints, ate)
        metrics['ate_mean_int_length']: calc_mean_interval_length(ate_conf_ints)

    return metrics


def calculate_ite_metrics(w_test: np.ndarray, t_test: np.ndarray, y0: np.ndarray, y1: np.ndarray,
                          ite: np.ndarray, fitted_estimators: List[BaseIteEstimator], goal="continuous",
                          meta_est_name='', dataname='', confound_col=''):
    ite_estimates = np.stack([fitted_estimator.estimate_ite(w_test) for
                              fitted_estimator in fitted_estimators],
                             axis=STACK_AXIS)

    if ite_estimates.ndim != 2:
        ite_estimates = ite_estimates.squeeze(axis=2)
        print('new shape is:', ite_estimates.shape)

    # Calculated for each unit/individual, this is the a vector of num units

    mean_ite_estimate = ite_estimates.mean(axis=STACK_AXIS)
    ite_bias = mean_ite_estimate - ite
    ite_abs_bias = np.abs(ite_bias)
    ite_squared_bias = ite_bias**2
    ite_variance = calc_vector_variance(ite_estimates, mean_ite_estimate)
    ite_std_error = np.sqrt(ite_variance)
    ite_mse = calc_vector_mse(ite_estimates, ite)
    ite_rmse = np.sqrt(ite_mse)

    # Calculated for a single dataset, so this is a vector of num datasets
    pehe_squared = calc_vector_mse(ite_estimates, ite, reduce_axis=(1 - STACK_AXIS))
    pehe = np.sqrt(pehe_squared)
    smape,ape = calc_smape(ite_estimates, ite, reduce_axis=(1 - STACK_AXIS))

    baseline_est = ite_estimates.mean(axis=(1 - STACK_AXIS))
    baseline_est = np.reshape(baseline_est, (-1,1))
    baseline_est = np.repeat(baseline_est,ite.shape[0],axis=(1-STACK_AXIS))
    baseline_bias = baseline_est - ite
    baseline_abs_bias = np.abs(baseline_bias)
    baseline_mse = calc_vector_mse(baseline_est, ite)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_pehe_squared = calc_vector_mse(baseline_est, ite, reduce_axis=(1 - STACK_AXIS))
    baseline_pehe = np.sqrt(baseline_pehe_squared)
    baseline_smape, baseline_ape = calc_smape(baseline_est, ite, reduce_axis=(1 - STACK_AXIS))

    qini_coeff, AUCs, rnd_AUCs, scores = calc_qini(ite_estimates, ite, w_test, t_test, y0, y1, goal)
    pol_mapes, pol_diffs = policy_mape(ite_estimates, ape, baseline_ape, w_test, goal)
    df = pd.DataFrame({"qini_coeff": qini_coeff, "AUCs": AUCs, "rnd_AUCs": rnd_AUCs, "scores": scores, "pol_mapes": pol_mapes, "pol_diffs" : pol_diffs})

    # df.to_csv('../results/' + f"qini_{meta_est_name}_{dataname}_withdiffs.csv", index=False)


    df.to_csv('../results/'+f"qini_{meta_est_name}_{dataname}_{confound_col}.csv", index=False)
    pd.DataFrame(ape).to_csv('../results/'+f"mape_{meta_est_name}_{dataname}_{confound_col}.csv", index=False)
    pd.DataFrame(baseline_ape).to_csv('../results/'+f"mape_{meta_est_name}_{dataname}_b_{confound_col}.csv", index=False)
    print('ite metrics done!')

    # TODO: ITE coverage
    # ate_coverage = calc_coverage(ate_conf_ints, ate)
    # ate_mean_int_length = calc_mean_interval_length(ate_conf_ints)

    return {
        'ite_bias': ite_bias,
        'ite_abs_bias': ite_abs_bias,
        'ite_squared_bias': ite_squared_bias,
        'ite_variance': ite_variance,
        'ite_std_error': ite_std_error,
        'ite_mse': ite_mse,
        'ite_rmse': ite_rmse,
        # 'ite_coverage': ite_coverage,
        # 'ite_mean_int_length': ite_mean_int_length,
        'pehe_squared': pehe_squared,
        'pehe': pehe,
        'smape': smape,
        'baseline_bias' : baseline_bias,
        'baseline_abs_bias' : baseline_abs_bias,
        'baseline_mse' : baseline_mse,
        'baseline_rmse' : baseline_rmse,
        'baseline_pehe_squared' : baseline_pehe_squared,
        'baseline_pehe': baseline_pehe,
        'baseline_smape': baseline_smape,
    }

def calculate_pred_model(gen_model: BaseGenModel, model,
                      n_seeds: int, conf_ints=True, return_ite_vectors=False,
                      ate=None, ite=None, goal='continuous'):
    if ate is None:
        ate = gen_model.ate()
    # else:
    #     print('Already computed ate')
    if ite is None:
        ite = gen_model.ite().squeeze()
    # else:
    #     print('Already computed ite')
    fitted_estimators = []
    estimator = model
    # print(estimator.econml_estimator.min_samples_leaf)
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed)
        # w = w[:100,:]
        # t = t[:100,:]
        # y = y[:100,:]
        w = np.concatenate((w,t),axis=1-STACK_AXIS)
        estimator.fit(w, y)
        print('fitting done!')
        fitted_estimators.append(deepcopy(estimator))

    w_test, t_test, (y0, y1) = gen_model.sample(seed=n_seeds+1, ret_counterfactuals=True)
    w_test = np.concatenate((w_test, t_test), axis=1-STACK_AXIS)
    estimates = np.stack([fitted_estimator.predict(w_test) for
                              fitted_estimator in fitted_estimators],
                             axis=STACK_AXIS)
    qini_coeff, AUCs, rnd_AUCs, scores = calc_qini(estimates, ite, w_test, t_test, y0, y1, goal)
    df = pd.DataFrame({"qini_coeff": qini_coeff, "AUCs": AUCs, "rnd_AUCs": rnd_AUCs, "scores": scores})
    df.to_csv('qini_rf_19_bad.csv', index=False)


    return 0



def calc_qini(estimates, target, w, t, y0,y1,goal):
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    qini_coeff = []
    AUCs = []
    rnd_AUCs = []
    scores = []
    df = pd.DataFrame(w)
    df['y0'] = y0
    df['y1'] = y1
    df['treatment'] = t
    df['ite'] = target
    for i in range(estimates.shape[STACK_AXIS]):
        qini = [0]
        df['estimates'] = estimates[i]
        for n in percentages:
            num = int(len(df) * (n / 100))
            if goal == "continuous":
                top_n = df.nsmallest(num, 'estimates')
            if goal == "binary":
                top_n = df.nlargest(num, 'estimates')
            n_treated = top_n[top_n['treatment'] == 1].shape[0]
            n_control = top_n[top_n['treatment'] == 0].shape[0]
            # scale_factor = n_treated / n_control

            # smape, ape = calc_smape(np.array(top_n['estimates']).reshape(-1,1), np.array(top_n['ite']), reduce_axis=(1 - STACK_AXIS))
            # smapes.extend(smape)

            # treated = top_n[top_n['treatment'] == 1]['ite'].sum()
            # control = top_n[top_n['treatment'] == 0]['ite'].sum()
            treated = top_n['y1'].sum()
            control = top_n['y0'].sum()

            # reduction = (scale_factor * control) - treated
            # reduction = treated - (scale_factor * control)
            if goal == "continuous":
                reduction = control - treated
            if goal == "binary":
                reduction = treated - control
            qini.append(reduction)

        scores.append(qini)

        qini = qini / max(abs(np.array(qini)))
        auc = np.trapz(qini, [0] + percentages)
        rnd_auc = qini[10]*100 / 2

        AUCs.append(auc)
        rnd_AUCs.append(rnd_auc)
        qini_coeff.append(auc - rnd_auc)

    return qini_coeff, AUCs, rnd_AUCs, scores

def policy_mape(estimates, ape, baseline_ape, w, goal):
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df = pd.DataFrame(w)
    scores = []
    diffs = []
    for i in range(estimates.shape[STACK_AXIS]):
        smapes = [0]
        diff = [0]
        df['estimates'] = estimates[i]
        df['ape'] = ape[i]
        df['ape_baseline'] = baseline_ape[i]
        for n in percentages:
            num = int(len(df) * (n / 100))
            if goal == "continuous":
                top_n = df.nsmallest(num, 'estimates')
            if goal == "binary":
                top_n = df.nlargest(num, 'estimates')
            smapes.append(top_n['ape'].mean())
            diff.append(top_n['ape'].mean() - top_n['ape_baseline'].mean())

        scores.append(smapes)
        diffs.append(diff)

    return scores, diffs

def calc_variance(estimates, mean_estimate):
    return calc_mse(estimates, mean_estimate)


def calc_mse(estimates, target):
    if isinstance(estimates, (list, tuple)):
        estimates = np.array(estimates)
    return ((estimates - target) ** 2).mean()

def calc_maape(estimates: np.ndarray, target: np.ndarray, reduce_axis=STACK_AXIS):
    assert isinstance(estimates, np.ndarray) and estimates.ndim == 2
    assert isinstance(target, np.ndarray) and target.ndim == 1
    assert target.shape[0] == estimates.shape[1 - STACK_AXIS]

    n_seeds = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_seeds, axis=STACK_AXIS)
    aape = np.arctan(abs((estimates - target) / (target)))
    nans = np.isnan(aape)
    aape[nans] = 0
    infs = np.isinf(aape)
    aape[infs] = 1.57
    maape = aape.mean(axis=reduce_axis) * 100

    return maape

def calc_smape(estimates: np.ndarray, target: np.ndarray, reduce_axis=STACK_AXIS):
    assert isinstance(estimates, np.ndarray) and estimates.ndim == 2
    assert isinstance(target, np.ndarray) and target.ndim == 1
    # estimates = np.transpose(estimates)
    assert target.shape[0] == estimates.shape[1 - STACK_AXIS]

    n_seeds = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_seeds, axis=STACK_AXIS)
    ape = 2 * abs(estimates - target) / (abs(target)+abs(estimates))
    nans = np.isnan(ape)
    ape[nans] = 0
    infs = np.isinf(ape)
    ape[infs] = 2
    # np.savetxt("mape.csv", ape, delimiter=",")

    smape = ape.mean(axis=reduce_axis) * 100
    return smape, ape




def calc_coverage(intervals: List[tuple], estimand):
    n_covers = sum(1 for interval in intervals if interval[0] <= estimand <= interval[1])
    return n_covers / len(intervals)


def calc_mean_interval_length(intervals: List[tuple]):
    return mean(interval[1] - interval[0] for interval in intervals)


def calc_vector_variance(estimates: np.ndarray, mean_estimate: np.ndarray):
    return calc_vector_mse(estimates, mean_estimate)


def calc_vector_mse(estimates: np.ndarray, target: np.ndarray, reduce_axis=STACK_AXIS):
    assert isinstance(estimates, np.ndarray) and estimates.ndim == 2
    assert isinstance(target, np.ndarray) and target.ndim == 1
    assert target.shape[0] == estimates.shape[1 - STACK_AXIS]

    n_seeds = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_seeds, axis=STACK_AXIS)
    return ((estimates - target) ** 2).mean(axis=reduce_axis)
