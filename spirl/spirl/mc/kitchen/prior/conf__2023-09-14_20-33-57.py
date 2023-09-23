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
    'epoch_cycles_train': 10,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_layers=5,
    hidden_size=128,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 2
data_config.dataset_spec.min_subseq_len = 10
data_config.dataset_spec.max_subseq_len = 30
data_config.dataset_spec.codebook_dir = "codebook/codebook.pickle"

