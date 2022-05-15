import causalml.inference.nn

from causal_estimators.base import BaseCausalMLEstimator

class CEVAE(BaseCausalMLEstimator):

    def __init__(self, outcome_dist='normal',
                      latent_dim=10,
                      hidden_dim=50,
                      num_epochs=100,
                      batch_size=100,
                      learning_rate=0.001,
                      learning_rate_decay=0.01,
                      num_layers=5):
        super().__init__(causalml.inference.nn.CEVAE(outcome_dist=outcome_dist,
                      latent_dim=latent_dim,
                      hidden_dim=hidden_dim,
                      num_epochs=num_epochs,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      num_layers=num_layers))