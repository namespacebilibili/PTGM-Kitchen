from spirl.configs.hrl.kitchen.base_conf import *

from spirl.configs.hrl.kitchen.base_conf import *
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import CodebookBasedActionPriorSACAgent
from spirl.models.bc_mdl import Steve
from spirl.models.steve_prior import Prior
from spirl.configs.default_data_configs.kitchen import data_spec_prior
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.rl.agents.skill_space_agent import GoalBasedAgent
from spirl.data.kitchen.src.kitchen_data_loader import get_codebook
# from spirl.data.kitchen.src.kitchen_data_loader import get_full_codebook

# update policy to use prior model for computing divergence
prior_model_config = AttrDict(
    state_dim=data_spec_prior.state_dim,
    action_dim=50,
    n_layers=5,
    hidden_size=128,
    model_checkpoint = '/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/prior_50/weights/weights_ep4.pth',
)

ll_model_params = AttrDict(
    state_dim = data_spec.state_dim,
    input_dim = 60,
    goal_dim = 21,
    action_dim=data_spec.n_actions,
    n_layers=5,
    hidden_size=128,
)

hl_policy_params = AttrDict(
    action_dim=50,      
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,
)
# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=hl_policy_params.action_dim,
    n_layers=5,  # number of policy network laye
    nz_mid=256,
    action_input=False,
)

hl_policy_params.update(AttrDict(
    prior_model=Prior, 
    prior_model_params = prior_model_config,
    prior_model_checkpoint = prior_model_config.model_checkpoint,
))

hl_agent_config.update(AttrDict(
    policy= LearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
))
# update agent, set target divergence
agent_config.hl_agent = CodebookBasedActionPriorSACAgent
agent_config.hl_agent_params.update(AttrDict(
    td_schedule_params=AttrDict(p=5.),
    codebook = get_codebook,
))

ll_agent_config.update(AttrDict(
    model=Steve,
    model_params=ll_model_params,
    model_checkpoint='experiments/mc/kitchen/steve/20_21_steve/weights/weights_ep80.pth',
))

hl_policy_params.update(AttrDict(
    action_dim = data_spec_prior.n_actions, 
))

agent_config.update(AttrDict(
    hl_interval = 10,
    ll_agent = GoalBasedAgent,
    ll_agent_params = ll_agent_config,
))

# import pickle
# with open("test_conf.pkl",'wb') as f:
#     pickle.dump(ll_agent_config, f)

# import pickle
# with open("test_prior.pkl",'wb') as f:
#     pickle.dump(prior_model_config, f)