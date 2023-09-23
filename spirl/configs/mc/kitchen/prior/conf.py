import os

from spirl.models.steve_prior import Prior
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.kitchen import data_spec_prior
from spirl.components.evaluator import DummyEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

data_spec = data_spec_prior
configuration = {
    'model': Prior,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 50,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=50,
    n_layers=5,
    hidden_size=128,
    mode='train',
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 11
data_config.dataset_spec.codebook_dir = 'spirl/codebook/codebook_50.pickle'
