from comet_ml import Experiment
import argparse
import os
import numpy as np
import torch
import gpytorch
from data.lalonde import load_lalonde
from data.lbidd import load_lbidd
from data.ihdp import load_ihdp
from data.twins import load_twins
from data.bpic17 import load_bpic
from models import TarNet, preprocess, TrainingParams, MLPParams, LinearModel, GPModel, TarGPModel, GPParams
from models import distributions
import helpers
from collections import OrderedDict
import json
# from utils import get_duplicates

######### CASE AND RESOURCE ATTRIBUTES FOR TRAINING GENERATIVE MODEL ################
elim_17 = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
           39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
           64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
           89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
           111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
           131, 132, 133, 134, 135, 136, 137, 138, 0,1,154, 155, 156, 157, 158, 162, 163, 164, 165, 166, 167,
           168, 169, 170, 171, 172, 173,174]
elim_19 = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
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
           407, 408, 409, 410, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
           434, 435, 436,437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453,
           454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473,
           474, 475, 476, 477, 478, 479, 480, 481]


def get_data(args):
    data_name = args.data.lower()
    ate = None
    ites = None
    if data_name == "lalonde" or data_name == "lalonde_psid" or data_name == "lalonde_psid1":
        w, t, y = load_lalonde(obs_version="psid", dataroot=args.dataroot)
    elif data_name == "lalonde_rct":
        w, t, y = load_lalonde(rct=True, dataroot=args.dataroot)
    elif data_name == "lalonde_cps" or data_name == "lalonde_cps1":
        w, t, y = load_lalonde(obs_version="cps", dataroot=args.dataroot)
    elif data_name.startswith("lbidd"):
        # Valid string formats: lbidd_<link>_<n> and lbidd_<link>_<n>_counterfactual
        # Valid <link> options: linear, quadratic, cubic, exp, and log
        # Valid <n> options: 1k, 2.5k, 5k, 10k, 25k, and 50k
        options = data_name.split("_")
        link = options[1]
        n = options[2]
        observe_counterfactuals = (len(options) == 4) and (options[3] == "counterfactual")
        d = load_lbidd(n=n, observe_counterfactuals=observe_counterfactuals, link=link,
                       dataroot=args.dataroot, return_ate=True, return_ites=True)
        ate = d["ate"]
        ites = d['ites']
        if observe_counterfactuals:
            w, t, y = d["obs_counterfactual_w"], d["obs_counterfactual_t"], d["obs_counterfactual_y"]
        else:
            w, t, y = d["w"], d["t"], d["y"]
    elif data_name == "ihdp":
        d = load_ihdp(return_ate=True, return_ites=True)
        w, t, y, ate, ites = d["w"], d["t"], d["y"], d['ate'], d['ites']
    elif data_name == "ihdp_counterfactual":
        d = load_ihdp(observe_counterfactuals=True)
        w, t, y = d["w"], d["t"], d["y"]
    elif data_name == "twins":
        d = load_twins(dataroot=args.dataroot)
        w, t, y = d["w"], d["t"], d["y"]
    elif data_name == "bpic16":
        w, t, y = load_bpic(dataroot=args.dataroot, year=16)
    elif data_name == "bpic17":
        w, t, y = load_bpic(dataroot=args.dataroot, year=17)
        # w = w[:, elim_17]
    elif data_name == "bpic19":
        w, t, y = load_bpic(dataroot=args.dataroot, year=19)
        # w = w[:, elim_19]
    elif data_name == "bpic20":
        w, t, y = load_bpic(dataroot=args.dataroot, year=20)
    elif data_name == "bpic17c":
        w, t, y = load_bpic(dataroot=args.dataroot, year=17)
        w = w[:, elim_17]
    elif data_name == "bpic19c":
        w, t, y = load_bpic(dataroot=args.dataroot, year=19)
        w = w[:, elim_19]

        #w, t, y = d["w"], d["t"], d["y"]


    else:
        raise (Exception("dataset {} not implemented".format(args.data)))

    return ites, ate, w, t, y


def get_distribution(args):
    """
    args.dist_args should be a list of keyward:value pairs.

      examples:
      1) ['ndim:5']
      2) ['ndim:10', 'base_distribution:uniform']
    """
    dist_name = args.dist
    kwargs = dict()
    if len(args.dist_args) > 0:
        for a in args.dist_args:
            k, v = a.split("=")
            if v.isdigit():
                v = int(v)
            kwargs.update({k: v})

    if dist_name in distributions.BaseDistribution.dist_names:
        dist = distributions.BaseDistribution.dists[dist_name](**kwargs)
    else:
        raise NotImplementedError(
            f"Got dist argument `{dist_name}`, not one of {distributions.BaseDistribution.dist_names}"
        )
    if args.atoms:
        dist = distributions.MixedDistribution(args.atoms, dist)
    return dist


def evaluate(args, model):
    all_runs = list()
    t_pvals = list()
    y_pvals = list()

    for _ in range(args.num_univariate_tests):
        uni_metrics = model.get_univariate_quant_metrics(dataset="test")
        all_runs.append(uni_metrics)
        t_pvals.append(uni_metrics["t_ks_pval"])
        y_pvals.append(uni_metrics["y_ks_pval"])

    summary = OrderedDict()

    summary.update(nll=model.best_val_loss)
    summary.update(avg_t_pval=sum(t_pvals) / args.num_univariate_tests)
    summary.update(avg_y_pval=sum(y_pvals) / args.num_univariate_tests)
    summary.update(min_t_pval=min(t_pvals))
    summary.update(min_y_pval=min(y_pvals))
    summary.update(q30_t_pval=np.percentile(t_pvals, 30))
    summary.update(q30_y_pval=np.percentile(y_pvals, 30))
    summary.update(q50_t_pval=np.percentile(t_pvals, 50))
    summary.update(q50_y_pval=np.percentile(y_pvals, 50))

    summary.update(ate_exact=model.ate().item())
    summary.update(ate_noisy=model.noisy_ate().item())

    return summary, all_runs


def main(args, save_args=True, log_=True):
    # create logger
    helpers.create(*args.saveroot.split("/"))
    logger = helpers.Logging(args.saveroot, "log.txt", log_)
    logger.info(args)

    # save args
    if save_args:
        with open(os.path.join(args.saveroot, "args.txt"), "w") as file:
            file.write(json.dumps(args.__dict__, indent=4))

    # dataset
    logger.info(f"getting data: {args.data}")
    ites, ate, w, t, y = get_data(args)

    # comet logging
    if args.comet:
        exp = Experiment(project_name="causal-benchmark", auto_metric_logging=False)
        exp.add_tag(args.data)
        logger.info(f"comet url: {exp.url}")
    else:
        exp = None

    logger.info(f"ate: {ate}")

    # distribution of outcome (y)
    distribution = get_distribution(args)
    logger.info(distribution)

    # training params
    training_params = TrainingParams(
        lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs
    )
    logger.info(training_params.__dict__)

    # initializing model
    w_transform = preprocess.Preprocess.preps[args.w_transform]
    y_transform = preprocess.Preprocess.preps[args.y_transform]
    outcome_min = 0 if args.y_transform == "Normalize" else None
    outcome_max = 1 if args.y_transform == "Normalize" else None

    # model type
    additional_args = dict()
    if args.model_type == 'tarnet':
        Model = TarNet

        logger.info('model type: tarnet')
        mlp_params = MLPParams(
            n_hidden_layers=args.n_hidden_layers,
            dim_h=args.dim_h,
            activation=getattr(torch.nn, args.activation)(),
        )
        logger.info(mlp_params.__dict__)
        network_params = dict(
            mlp_params_w=mlp_params,
            mlp_params_t_w=mlp_params,
            mlp_params_y0_w=mlp_params,
            mlp_params_y1_w=mlp_params,
        )
    elif args.model_type == 'linear':
        Model = LinearModel

        logger.info('model type: linear model')
        network_params = dict()
    elif 'gp' in args.model_type:
        if args.model_type == 'gp':
            Model = GPModel
        elif args.model_type == 'targp':
            Model = TarGPModel
        else:
            raise Exception(f'model type {args.model_type} not implemented')
        logger.info('model type: linear model')

        kernel_t = gpytorch.kernels.__dict__[args.kernel_t]()
        kernel_y = gpytorch.kernels.__dict__[args.kernel_y]()
        var_dist = gpytorch.variational.__dict__[args.var_dist]
        network_params = dict(
            gp_t_w=GPParams(kernel=kernel_t, var_dist=var_dist),
            gp_y_tw=GPParams(kernel=kernel_y, var_dist=None),
        )
        logger.info(f'gp_t_w: {repr(network_params["gp_t_w"])}'
                    f'gp_y_tw: {repr(network_params["gp_y_tw"])}')
        additional_args['num_tasks'] = args.num_tasks
    else:
        raise Exception(f'model type {args.model_type} not implemented')

    if args.n_hidden_layers < 0:
        raise Exception(f'`n_hidden_layers` must be nonnegative, got {args.n_hidden_layers}')

    print(args.model_type)
    model = Model(w, t, y,
                  training_params=training_params,
                  network_params=network_params,
                  binary_treatment=True, outcome_distribution=distribution,
                  outcome_min=outcome_min,
                  outcome_max=outcome_max,
                  train_prop=args.train_prop,
                  val_prop=args.val_prop,
                  test_prop=args.test_prop,
                  seed=args.seed,
                  early_stop=args.early_stop,
                  patience=args.patience,
                  ignore_w=args.ignore_w,
                  grad_norm=args.grad_norm,
                  w_transform=w_transform, y_transform=y_transform,  # TODO set more args
                  savepath=os.path.join(args.saveroot, 'model.pt'),
                  test_size=args.test_size,
                  additional_args=additional_args)
    
    # TODO GPU support
    if args.train:
        model.train(print_=logger.info, comet_exp=exp)

    # evaluation
    if args.eval:
        summary, all_runs = evaluate(args, model)
        logger.info(summary)
        with open(os.path.join(args.saveroot, "summary.txt"), "w") as file:
            file.write(json.dumps(summary, indent=4))
        with open(os.path.join(args.saveroot, "all_runs.txt"), "w") as file:
            file.write(json.dumps(all_runs))

        model.plot_ty_dists()

    return model


def get_args():
    parser = argparse.ArgumentParser(description="causal-gen")

    # dataset
    parser.add_argument("--data", type=str, default="bpic19")  # TODO: fix choices
    parser.add_argument(
        "--dataroot", type=str, default="datasets"
    )  # TODO: do we need it?
    parser.add_argument("--saveroot", type=str, default="save")
    parser.add_argument("--train", type=eval, default=True, choices=[True, False])
    parser.add_argument("--eval", type=eval, default=True, choices=[True, False])
    parser.add_argument('--overwrite_reload', type=str, default='',
                        help='secondary folder name of an experiment')  # TODO: for model loading

    # model type
    parser.add_argument('--model_type', type=str, default='tarnet',
                        choices=['tarnet', 'linear', 'gp', 'targp'])  # TODO: renaming tarnet to be dragonnet

    # distribution of outcome (y)
    parser.add_argument('--dist', type=str, default='Bernoulli',
                        choices=distributions.BaseDistribution.dist_names)
    parser.add_argument("--dist_args", type=str, default=list(), nargs="+")
    parser.add_argument("--atoms", type=float, default=list(), nargs="+")

    # architecture for tarnet
    parser.add_argument("--n_hidden_layers", type=int, default=1)
    parser.add_argument("--dim_h", type=int, default=64)
    parser.add_argument("--activation", type=str, default="ReLU")

    # architecture for gp
    parser.add_argument("--kernel_t", type=str, default="RBFKernel",
                        choices=gpytorch.kernels.__all__)
    parser.add_argument("--kernel_y", type=str, default="RBFKernel",
                        choices=gpytorch.kernels.__all__)
    parser.add_argument("--var_dist", type=str, default="MeanFieldVariationalDistribution",
                        choices=[vd for vd in gpytorch.variational.__all__ if 'VariationalDistribution' in vd])
    parser.add_argument("--num_tasks", type=int, default=32,
                        help='number of latent variables for the GP atom softmax classifier')

    # training params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=eval, default=True, choices=[True, False])
    parser.add_argument("--patience", type=int)

    parser.add_argument("--ignore_w", type=eval, default=False, choices=[True, False])
    parser.add_argument("--grad_norm", type=float, default=float("inf"))
    parser.add_argument("--test_size", type=int)

    parser.add_argument('--w_transform', type=str, default='Standardize',
                        choices=preprocess.Preprocess.prep_names)
    parser.add_argument('--y_transform', type=str, default='Normalize',
                        choices=preprocess.Preprocess.prep_names)
    parser.add_argument("--train_prop", type=float, default=0.5)
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--test_prop", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--comet", type=eval, default=False, choices=[True, False])

    # evaluation
    parser.add_argument("--num_univariate_tests", type=int, default=100)

    return parser


if __name__ == "__main__":
    main(get_args().parse_args())
