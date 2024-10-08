from spirl.utils.general_utils import AttrDict
from spirl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset
from spirl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset_prior


data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=9,
    state_dim=60,
    goal_dim=21,
    env_name="kitchen-mixed-v0",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280

data_spec_prior = AttrDict(
    dataset_class=D4RLSequenceSplitDataset_prior,
    n_actions=50,
    state_dim=60,
    env_name="kitchen-mixed-v0",
    res=128,
    crop_rand_subseq=True,
)
data_spec_prior.max_seq_len = 280
