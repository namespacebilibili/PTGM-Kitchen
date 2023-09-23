import gym
import d4rl
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from spirl.models.bc_mdl import Steve
from spirl.models.steve_prior import Prior
from spirl.rl.agents.skill_space_agent import GoalBasedAgent
from spirl.utils.general_utils import AttrDict
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.components.checkpointer import CheckpointHandler
# ll_model_params = AttrDict(
#     state_dim = data_spec.state_dim,
#     input_dim = 60,
#     goal_dim = 60,
#     action_dim=data_spec.n_actions,
#     n_layers=5,
#     hidden_size=128,
# )
# ll_agent_config.update(AttrDict(
#     model=Steve,
#     model_params=ll_model_params,
#     model_checkpoint='/home/yhq/Desktop/spirl-master/steve/weights_ep179.pth',
# ))
# hp = {'batch_size': 256, 
#     'replay': 'spirl.rl.components.replay_buffer.UniformReplayBuffer', 'replay_params': {}, 
#     'clip_q_target': False, 'model': 'spirl.models.bc_mdl.Steve', 
#     'model_params': {'state_dim': 60, 'input_dim': 60, 'goal_dim': 60, 'action_dim': 9, 'n_layers': 5, 'hidden_size': 128, 'device' : 'cuda'}, 
#     'model_checkpoint': '/home/yhq/Desktop/spirl-master/steve/weights_ep179.pth',}
useful_idx = np.array([0,1,2,3,4,5,6,7,8,11,12,17,18,22,23,24,25,26,27,28,29])
with open("test_prior.pkl",'rb') as f:
    prior_hp = pickle.load(f)
# prior_hp.update(AttrDict(model_checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/20_21_prior/weights/weights_ep7.pth',
#                 checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/20_21_prior/weights/weights_ep7.pth'))
prior_hp.update(AttrDict(model_checkpoint=None,
                checkpoint=None))
# print(f"prior_hp = {prior_hp}")
# ckpt = torch.load('/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/steve/20_21_steve/weights/weights_ep10.pth')
# print(ckpt['state_dict'].keys())
prior = Prior(prior_hp)
checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/20_21_prior/weights/weights_ep26.pth'
CheckpointHandler.load_weights(checkpoint, model=prior)



with open("test_conf.pkl",'rb') as f:
    hp = pickle.load(f)
hp.update(AttrDict(checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/steve/20_21_steve/weights/weights_ep10.pth'))
model = GoalBasedAgent(hp)
# env = gym.make("kitchen-mixed-v0")
env_config = AttrDict(
    reward_norm=1.,
)
env = KitchenEnv(env_config)
with open("codebook/codebook_robot20_full.pickle",'rb') as f:
    codes = pickle.load(f)
# code = codes[1]
# inputs = AttrDict(states=torch.tensor([code],dtype=torch.float))
# output = prior(inputs)
# hl_choose = torch.argmax(torch.squeeze(output.pred_act.detach())).item()
# print(hl_choose)
state = env.reset()
# print(f"state = {state}")
# print(code)
# inputs = AttrDict(states = torch.tensor([state],dtype=torch.float), goals = torch.tensor([code],dtype=torch.float))
# wonderful_seq = [18,14,5,0,7,12,13,]
hl_choose = 0
last_hl = -1
for i in range(500):
    if i % 30 == 0:
        inputs = AttrDict(states=torch.tensor([state],dtype=torch.float))
        output = prior(inputs)
        output = output.pred_act.detach()
        output = torch.nn.functional.softmax(output)
        # print(output)
        output = torch.multinomial(output, 1, replacement=False)
        hl_choose = torch.squeeze(output).item()
        if last_hl != -1 and last_hl == hl_choose:
            hl_choose = random.randint(0,19)
        # if i <= 120 and i % 10 == 0:
        #     hl_choose = wonderful_seq[(i//20)]
        print(hl_choose)
        last_hl = hl_choose
        # print(f"state = {state[useful_idx]}")
        # print(f"target = {codes[hl_choose][useful_idx]}")
    
    obs = np.concatenate((np.array([state]), np.array([codes[hl_choose][useful_idx]])), axis=1)
    step = model._act(obs)
    action = torch.squeeze(step.action).detach()
    state, reward, done, info = env.step(action)
    if i % 10 == 0:
        img = env.render(mode="rgb_array")
        plt.imshow(img)
        plt.savefig(f"img/video_{i//10}.jpg")
# env = gym.make("kitchen-mixed-v0")
# env.set_state(qpos=code[:30],qvel=env.sim.model.key_qvel[0])
# img = env.render(mode="rgb_array")
# plt.imshow(img)
# plt.savefig(f"img/goal.jpg")


# for code in codes:
#     env.set_state(qpos=code[:30],qvel=env.sim.model.key_qvel[0])
#     img = env.render(mode="rgb_array")
#     plt.imshow(img)
#     plt.savefig(f"robot_code20_{idx}.png")
# self.init_qpos = self.sim.model.key_qpos[0].copy()

# # For the microwave kettle slide hinge
# self.init_qpos = np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
#                             2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
#                             3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
#                            -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
#                             4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
#                            -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
#                             3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
#                            -6.62095997e-03, -2.68278933e-04])

# self.init_qvel = self.sim.model.key_qvel[0].copy()


# def set_state(self, qpos, qvel):
#     assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
#     state = self.sim.get_state()
#     for i in range(self.model.nq):
#         state.qpos[i] = qpos[i]
#     for i in range(self.model.nv):
#         state.qvel[i] = qvel[i]
#     self.sim.set_state(state)
#     self.sim.forward()

