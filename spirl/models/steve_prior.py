from contextlib import contextmanager
import torch
import torch.nn as nn
import copy

from spirl.components.base_model import BaseModel
from spirl.modules.losses import NLL
from spirl.modules.losses import CELoss
from spirl.modules.subnetworks import Predictor, Encoder
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.utils.pytorch_utils import RemoveSpatial, ResizeSpatial
from spirl.modules.variational_inference import ProbabilisticModel, MultivariateGaussian
from spirl.modules.layers import LayerBuilderParams
from spirl.modules.mdn import GMM
from spirl.utils.pytorch_utils import no_batchnorm_update

class Prior(BaseModel):
    """Prior model for Our Method (steve1)"""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self.device = self._hp.device
        self.build_network()
        self.load_weights_and_freeze()

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass

    def _default_hparams(self):
        default_dict = ParamDict({
            'use_convs': False,
            'device': 'cuda:0',
        }) 

        default_dict.update({
            'state_dim': 1,
            'action_dim': 1,
            'n_processing_layers': 3,   # number of layers in MLPs
            'mse_weight': 1.,    # weight of MSE reconstruction loss
            'output_type': 'flow', 
            'nz_mid': 128,              # number of dimensions for internal feature spaces
            'checkpoint': None,
            'mode': "test"
        })

        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        assert not self._hp.use_convs
        print(f"self._hp = {self._hp}, self.input_dim = {self.input_dim}, self.output_dim = {self.output_dim}")
        self.net = Predictor(self._hp, input_size=self.input_dim, output_size=self.output_dim)

    def forward(self, inputs):
        """Forward for Prior
        :arg inputs: dict:"states" [N, state_dim] "gt" [N,1]
        :return output: predict codebook [N, codebook_dim]
        """
        output = AttrDict()
        output.pred_act = self._compute_output_dist(self._net_inputs(inputs))
        return output

    def _compute_output_dist(self, inputs):
        if self._hp.output_type == 'gmm':
            return GMM(*self.net(inputs))
        elif self._hp.output_type == 'flow':
            if inputs.shape[0] == 1:
                with no_batchnorm_update(self.net):
                    output = self.net(inputs)
            else:
                output = self.net(inputs)
            return output if self._hp.mode == 'train' else output.softmax(dim=-1)
        else:
            return MultivariateGaussian(self.net(inputs))

    def compute_learned_prior(self, inputs, first_only=False):
        """Used in BC prior regularized RL policies."""
        assert first_only is True       # do not currently support ensembles for BC model
        if inputs.shape[0] == 1:
            with no_batchnorm_update(self.net):
                output = self.net(inputs)
        else:
            output = self.net(inputs)
        return output if self._hp.mode == 'train' else output.softmax(dim=-1)

    def loss(self, model_output, inputs):
        losses = AttrDict()
        # crossentropy loss
        losses.crossentropy = CELoss()(model_output.pred_act, self._regression_targets(inputs))
        losses.total = self._compute_total_loss(losses)
        return losses

    def _log_outputs(self, outputs, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        """Optionally visualizes outputs of Steve.
        :arg output: output of Steve model forward pass
        :arg inputs: dict with 'states', 'actions', 'images', 'goals' keys from data loader
        :arg losses: output of Steve model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        # self._logger.log_scalar(self.beta, "beta", step, phase)

        # log videos/gifs in tensorboard
        if log_images:
            print('{} {}: logging videos'.format(phase, step))
            self._logger.visualize(outputs, inputs, losses, step, phase, logger, **logging_kwargs)

    # def run(self, inputs):
    #     inputs = map2torch(inputs, device=self.device)
    #     action = self.net(self.get_input(inputs))
    #     return AttrDict(action=action[None]) # Warning

    def load_weights_and_freeze(self):
        if self._hp.checkpoint is not None:
            print("Loading model from {}!".format(self._hp.checkpoint))
            self.net.load_state_dict(torch.load(self._hp.checkpoint))
            self.net.to(self.device)

    def _net_inputs(self, inputs):
        states = inputs.states.reshape(-1, self._hp.state_dim)
        return states

    def _regression_targets(self, inputs):
        return inputs.gt.reshape(-1)

    @property
    def input_dim(self):
        return self._hp.state_dim

    @property
    def output_dim(self):
        return self._hp.action_dim

    @property
    def resolution(self):
        return 64       # return dummy resolution, images are not used by this model

class BCMdl(BaseModel):
    """Simple recurrent forward predictor network with image encoder and decoder."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self.device = self._hp.device

        self.build_network()

    def _default_hparams(self):
        # put new parameters in here:
        return super()._default_hparams().overwrite(ParamDict({
            'use_convs': False,
            'device': None,
            'state_dim': 1,             # dimensionality of the state space
            'action_dim': 1,            # dimensionality of the action space
            'nz_mid': 128,              # number of dimensions for internal feature spaces
            'n_processing_layers': 5,   # number of layers in MLPs
            'output_type': 'gauss',     # distribution type for learned prior, ['gauss', 'gmm', 'flow']
            'n_gmm_prior_components': 5,    # number of Gaussian components for GMM learned prior
        }))

    def build_network(self):
        assert not self._hp.use_convs   # currently only supports non-image inputs
        assert self._hp.output_type == 'gauss'  # currently only support unimodal output
        self.net = Predictor(self._hp, input_size=self._hp.state_dim, output_size=self._hp.action_dim * 2)

    def forward(self, inputs, use_learned_prior=False):
        """
        forward pass at training time
        """
        output = AttrDict()

        output.pred_act = self._compute_output_dist(self._net_inputs(inputs))
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.nll = NLL()(model_output.pred_act, self._regression_targets(inputs))

        losses.total = self._compute_total_loss(losses)
        return losses

    def _compute_output_dist(self, inputs):
        if self._hp.output_type == 'gmm':
            return GMM(*self.net(inputs))
        elif self._hp.output_type == 'flow':
            return self.net(inputs)
        else:
            return MultivariateGaussian(self.net(inputs))

    def _net_inputs(self, inputs):
        return inputs.states[:, 0]

    def _regression_targets(self, inputs):
        return inputs.actions[:, 0]

    def compute_learned_prior(self, inputs, first_only=False):
        """Used in BC prior regularized RL policies."""
        assert first_only is True       # do not currently support ensembles for BC model
        if len(inputs.shape) == 1:
            return self._compute_output_dist(inputs[None])[0]
        else:
            return self._compute_output_dist(inputs)

    @property
    def resolution(self):
        return 64  # return dummy resolution, images are not used by this model

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass


class ImageBCMdl(BCMdl):
    """Implements BC model with image input."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,            # input resolution
            'encoder_ngf': 8,           # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,        # number of input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed flat we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            input_nc=3*self._hp.n_input_frames,  # number of input feature maps
            ngf=self._hp.encoder_ngf,         # number of feature maps in shallowest level
            nz_enc=self._hp.nz_mid,     # size of image encoder output feature
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))

    def build_network(self):
        self.net = nn.Sequential(
            ResizeSpatial(self._hp.input_res),
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            Predictor(self._hp, input_size=self._hp.nz_mid, output_size=self._hp.action_dim * 2),
        )

    def _net_inputs(self, inputs):
        return inputs.images[:, :self._hp.n_input_frames]\
            .reshape(inputs.images.shape[0], -1, self.resolution, self.resolution)

    def _regression_targets(self, inputs):
        return inputs.actions[:, (self._hp.n_input_frames-1)]

    def unflatten_obs(self, raw_obs):
        """Utility to unflatten [obs, prior_obs] concatenated observation (for RL usage)."""
        #if len(raw_obs.shape) == 1:
        #    raw_obs = raw_obs[None]
        assert len(raw_obs.shape) == 2 and raw_obs.shape[1] == self._hp.state_dim \
            + self._hp.input_res**2 * 3 * self._hp.n_input_frames
        return AttrDict(
            obs=raw_obs[:, self._hp.state_dim],
            prior_obs=raw_obs[:, self._hp.state_dim:].reshape(raw_obs.shape[0], 3*self._hp.n_input_frames,
                                                              self._hp.input_res, self._hp.input_res)
        )

    @property
    def resolution(self):
        return self._hp.input_res
