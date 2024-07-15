# Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning

[[Project Website]](https://sites.google.com/view/ptgm-iclr/)[[Paper]](https://openreview.net/forum?id=o2IEmeLL9r)

This is the code base for experiments in the `Kitchen` env for our paper [Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning](https://openreview.net/forum?id=o2IEmeLL9r).
The model architecture and training code builds on a code base on [spirl](https://github.com/clvrai/spirl).

## Requirements

- python 3.7+
- mujoco 2.0 (for RL experiments)
- Ubuntu 18.04

## Installation Instructions

Create a virtual environment and install all required packages.
```
cd codebook4spirl
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies and package
pip3 install -r requirements.txt
pip3 install -e .
```

Set the environment variables that specify the root experiment and data directories. For example: 
```
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

Finally, install **spirl'sfork** of the [D4RL benchmark](https://github.com/kpertsch/d4rl) repository by following its installation instructions.
It will provide both, the kitchen environment as well as the training data for the skill prior model in kitchen and maze environment.

## Train your own model
All results will be written to [WandB](https://www.wandb.com/). Before running any of the commands below, 
create an account and then change the WandB entity and project name at the top of [train.py](spirl/train.py) and
[rl/train.py](spirl/rl/train.py) to match your account.

For dataset downloading, check [link](https://drive.google.com/file/d/1kWYy9L4w2lLXXzMH1bvUbk6NLFphMmDD/view?usp=drive_link) and put it under `spirl/codebook/`.

First you need to generate codebook using [generate_codebook.py](generate_codebook.py), codebook will be put at [spirl/codebook](spirl/codebook).You can adjust codebook size and code dim in [generate_codebook.py](generate_codebook.py), default parameter is 50 code in codebook and 21 dim for each code.

Then to train a prior model, run:
```
python3 spirl/train.py --path=spirl/configs/ptgm/kitchen/prior --val_data_size=160
```
The config is at [spirl/configs/mc/kitchen/prior/conf.py](spirl/configs/mc/kitchen/prior/conf.py). To match the codebook size, you should adjust codebook path, and `action_dim` of `data_spec_prior` in [spirl/configs/default_data_configs/kitchen.py][spirl/configs/default_data_configs/kitchen.py]

To train a low level policy (predict $a$ using $s$ and $g$), run:
```
python3 spirl/train.py --path=spirl/configs/ptgm/kitchen/steve --val_data_size=160
```
Default state dim is 60, action dim is 9 and goal dim is 21 (size of code in codebook). The config is at [spirl/configs/mc/kitchen/steve/conf.py](spirl/configs/mc/kitchen/steve/conf.py)

Finally for RL training, you can change the config and model path in [spirl/configs/hrl/kitchen/steve/conf.py](spirl/configs/hrl/kitchen/steve/conf.py), run:
```
python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/steve --seed=0 --prefix=SPIRL_kitchen_seed0
```
Don't forget change codebook path in `get_codebook` in [spirl/data/kitchen/src/kitchen_data_loader.py](spirl/data/kitchen/src/kitchen_data_loader.py).

## Starting to Modify the Code

### Modifying the hyperparameters
The default hyperparameters are defined in the respective model files, e.g. in [```skill_prior_mdl.py```](spirl/models/skill_prior_mdl.py#L47)
for the SPIRL model. Modifications to these parameters can be defined through the experiment config files (passed to the respective
command via the `--path` variable). For an example, see [```kitchen/hierarchical/conf.py```](spirl/configs/skill_prior_learning/kitchen/hierarchical/conf.py).


### Adding a new dataset for model training
All code that is dataset-specific should be placed in a corresponding subfolder in `spirl/data`. 
To add a data loader for a new dataset, the `Dataset` classes from [```data_loader.py```](spirl/components/data_loader.py) need to be subclassed
and the `__getitem__` function needs to be overwritten to load a single data sample. The output `dict` should include the following
keys:

```
dict({
    'states': (time, state_dim)                 # state sequence (for state-based prior inputs)
    'actions': (time, action_dim)               # action sequence (as skill input for training prior model)
    'images':  (time, channels, width, height)  # image sequence (for image-based prior inputs)
})
```

All datasets used with the codebase so far have been based on `HDF5` files. The `GlobalSplitDataset` provides functionality to read all
HDF5-files in a directory and split them in `train/val/test` based on percentages. The `VideoDataset` class provides
many functionalities for manipulating sequences, like randomly cropping subsequences, padding etc.

### Adding a new RL environment
To add a new RL environment, simply define a new environent class in `spirl/rl/envs` that inherits from the environment interface
in [```spirl/rl/components/environment.py```](spirl/rl/components/environment.py).


### Modifying the skill prior model architecture
Start by defining a model class in the `spirl/models` directory that inherits from the `BaseModel` or `SkillPriorMdl` class. 
The new model needs to define the architecture in the constructor (e.g. by overwriting the `build_network()` function), 
implement the forward pass and loss functions,
as well as model-specific logging functionality if desired. For an example, see [```spirl/models/skill_prior_mdl.py```](spirl/models/skill_prior_mdl.py).

Note, that most basic architecture components (MLPs, CNNs, LSTMs, Flow models etc) are defined in `spirl/modules` and can be 
conveniently reused for easy architecture definitions. Below are some links to the most important classes.

|Component        | File         | Description |
|:------------- |:-------------|:-------------|
| MLP | [```Predictor```](spirl/modules/subnetworks.py#L33) | Basic N-layer fully-connected network. Defines number of inputs, outputs, layers and hidden units. |
| CNN-Encoder | [```ConvEncoder```](spirl/modules/subnetworks.py#L66) | Convolutional encoder, number of layers determined by input dimensionality (resolution halved per layer). Number of channels doubles per layer. Returns encoded vector + skip activations. |
| CNN-Decoder | [```ConvDecoder```](spirl/modules/subnetworks.py#L145) | Mirrors architecture of conv. encoder. Can take skip connections as input, also versions that copy pixels etc. |
| Processing-LSTM | [```BaseProcessingLSTM```](spirl/modules/recurrent_modules.py#L70) | Basic N-layer LSTM for processing an input sequence. Produces one output per timestep, number of layers / hidden size configurable.|
| Prediction-LSTM | [```RecurrentPredictor```](spirl/modules/recurrent_modules.py#L241) | Same as processing LSTM, but for autoregressive prediction. |
| Mixture-Density Network | [```MDN```](spirl/modules/mdn.py#L10) | MLP that outputs GMM distribution. |
| Normalizing Flow Model | [```NormalizingFlowModel```](spirl/modules/flow_models.py#L9) | Implements normalizing flow model that stacks multiple flow blocks. Implementation for RealNVP block provided. |

### Adding a new RL algorithm
The core RL algorithms are implemented within the `Agent` class. For adding a new algorithm, a new file needs to be created in
`spirl/rl/agents` and [```BaseAgent```](spirl/rl/components/agent.py#L19) needs to be subclassed. In particular, any required
networks (actor, critic etc) need to be constructed and the `update(...)` function needs to be overwritten. For an example, 
see the SAC implementation in [```SACAgent```](spirl/rl/agents/ac_agent.py#L67).

The main codebook prior regularized RL algorithm is implemented in [```CodebookBasedActionPriorSACAgent```](spirl/rl/agents/prior_sac_agent.py).


## Detailed Code Structure Overview
```
spirl
  |- components            # reusable infrastructure for model training
  |    |- base_model.py    # basic model class that all models inherit from
  |    |- checkpointer.py  # handles storing + loading of model checkpoints
  |    |- data_loader.py   # basic dataset classes, new datasets need to inherit from here
  |    |- evaluator.py     # defines basic evaluation routines, eg top-of-N evaluation, + eval logging
  |    |- logger.py        # implements core logging functionality using tensorboardX
  |    |- params.py        # definition of command line params for model training
  |    |- trainer_base.py  # basic training utils used in main trainer file
  |
  |- configs               # all experiment configs should be placed here
  |    |- data_collect     # configs for data collection runs
  |    |- default_data_configs   # defines one default data config per dataset, e.g. state/action dim etc
  |    |- mc               # configs for our prior model and low-level policy
  |    |- hrl              # configs for hierarchical downstream RL
  |    |    |- steve       # configs for our method
  |    |- rl               # configs for non-hierarchical downstream RL
  |    |- skill_prior_learning   # configs for skill embedding and prior training (both hierarchical and flat)
  |
  |- data                  # any dataset-specific code (like data generation scripts, custom loaders etc)
  |- models                # holds all model classes that implement forward, loss, visualization
  |- modules               # reusable architecture components (like MLPs, CNNs, LSTMs, Flows etc)
  |- rl                    # all code related to RL
  |    |- agents           # implements core algorithms in agent classes, like SAC etc
  |    |- components       # reusable infrastructure for RL experiments
  |        |- agent.py     # basic agent and hierarchial agent classes - do not implement any specific RL algo
  |        |- critic.py    # basic critic implementations (eg MLP-based critic)
  |        |- environment.py    # defines environment interface, basic gym env
  |        |- normalization.py  # observation normalization classes, only optional
  |        |- params.py    # definition of command line params for RL training
  |        |- policy.py    # basic policy interface definition
  |        |- replay_buffer.py  # simple numpy-array replay buffer, uniform sampling and versions
  |        |- sampler.py   # rollout sampler for collecting experience, for flat and hierarchical agents
  |    |- envs             # all custom RL environments should be defined here
  |    |- policies         # policy implementations go here, MLP-policy and RandomAction are implemented
  |    |- utils            # utilities for RL code like MPI, WandB related code
  |    |- train.py         # main RL training script, builds all components + runs training
  |
  |- utils                 # general utilities, pytorch / visualization utilities etc
  |- train.py              # main model training script, builds all components + runs training loop and logging
```

The general philosophy is that each new experiment gets a new config file that captures all hyperparameters etc. so that experiments
themselves are version controllable.

## Troubleshooting

### Missing key 'completed_tasks' in Kitchen environment
Please make sure to install [our fork](https://github.com/kpertsch/d4rl) of the D4RL repository, **not** the original D4RL repository. We made a few small changes to the interface, which e.g. allow us to log the reward contributions for each of the subtasks separately.




