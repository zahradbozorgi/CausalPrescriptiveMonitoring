from collections import OrderedDict


HP = OrderedDict(
    # dataset
    data=['bpic17'],
    dataroot=['/home/zdashtbozorg/realcause/datasets/bpic17.csv'],  # TODO: MODIFY THIS PATH LOCALLY
    # saveroot=['save'],
    # train=[True],
    # eval=[True],
    # overwrite_reload=[''],

    # distribution of outcome (y)
    dist=['Bernoulli'],
    dist_args=[[]],  #,  ['ndim=10', 'base_distribution=normal'], ['ndim=1', 'base_distribution=uniform']
    #dist=['FactorialGaussian', 'Bernoulli', 'LogLogistic', 'LogNormal'],
    #dist_args=[[]],
    atoms=[[]],  # list of floats, or empty list

    # architecture
    n_hidden_layers=[2],
    dim_h=[64],
    activation=['ReLU'],

    # training params
    lr=[0.001],
    batch_size=[64],
    num_epochs=[100],
    early_stop=[True],
    ignore_w=[False],
    grad_norm=['inf'],

    w_transform=['Standardize'],
    y_transform=[[]], # 'Normalize'
    train_prop=[0.5],
    val_prop=[0.1],
    test_prop=[0.4],
    seed=[123],

    # evaluation
    num_univariate_tests=[100]
)
