import os

from spirl.models.bc_mdl import Steve
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.components.evaluator import DummyEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': Steve,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 50,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    goal_dim=data_spec.goal_dim,
    n_layers=5,
    hidden_size=128,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 11

