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

useful_idx = np.array([0,1,2,3,4,5,6,7,8,11,12,17,18,22,23,24,25,26,27,28,29])
with open("test_prior.pkl",'rb') as f:
    prior_hp = pickle.load(f)
prior_hp.update(AttrDict(action_dim=50))
# # prior_hp.update(AttrDict(model_checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/20_21_prior/weights/weights_ep7.pth',
# #                 checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/20_21_prior/weights/weights_ep7.pth'))
prior_hp.update(AttrDict(model_checkpoint=None,
                checkpoint=None))
# # print(f"prior_hp = {prior_hp}")
# # ckpt = torch.load('/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/steve/20_21_steve/weights/weights_ep10.pth')
# # print(ckpt['state_dict'].keys())
prior = Prior(prior_hp)
# checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/prior/prior_50/weights/weights_ep20.pth'
checkpoint='/home/yhq/Desktop/newcode_prior.pth'

CheckpointHandler.load_weights(checkpoint, model=prior)



with open("test_conf.pkl",'rb') as f:
    hp = pickle.load(f)
hp.update(AttrDict(checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/steve/20_21_steve/weights/weights_ep80.pth'))
hp.update(AttrDict(model_checkpoint='/home/yhq/Desktop/spirl-master/experiments/mc/kitchen/steve/20_21_steve/weights/weights_ep80.pth'))

# hp.update(AttrDict(checkpoint='/home/yhq/Desktop/newcode_steve.pth'))
# hp.update(AttrDict(model_checkpoint='/home/yhq/Desktop/newcode_steve.pth'))

print(hp)
model = GoalBasedAgent(hp)
# env = gym.make("kitchen-mixed-v0")
env_config = AttrDict(
    reward_norm=1.,
)
env = KitchenEnv(env_config)
with open("/home/yhq/Desktop/codebook_newfull50.pickle",'rb') as f:
    codes = pickle.load(f)
# code = codes[1]
# inputs = AttrDict(states=torch.tensor([code],dtype=torch.float))
# output = prior(inputs)
# hl_choose = torch.argmax(torch.squeeze(output.pred_act.detach())).item()
# print(hl_choose)
# env = gym.make("kitchen-mixed-v0")
# print(dir(env))
# print(env.reset.__code__.co_varnames)
# for k in range(10):
#     env.seed = k
#     kwargs = {'seed':k}
#     state = env.reset()
#     print(list(state))
#     img = env.render(mode="rgb_array")
#     plt.imshow(img)
#     plt.savefig(f"img/seed_{k}.jpg")
# print("done")
# print(f"state = {state}")
# print(code)
# inputs = AttrDict(states = torch.tensor([state],dtype=torch.float), goals = torch.tensor([code],dtype=torch.float))
# wonderful_seq = [18,14,5,0,7,12,13,]
hl_choose = 0
hl_chooses = [36,24,24,40,6,39,30,21,21,14,14,37,37,37,1,1,34,7,43,43,12,12,12,16,16,45,45]#37,37,1,34,35,7,7,43,43,16,16]
all_reward = []
for u in range(10):
    print(f"epoch {u}")
    ep_r = 0
    now_state = 0
    state = env.reset()
    idx = 0
    for i in range(280):
        if i % 10 == 0:
            inputs = AttrDict(states=torch.tensor([state],dtype=torch.float))
            output = prior(inputs)
            output = output.pred_act.detach()
            output = torch.nn.functional.softmax(output,dim=1)
            output = torch.multinomial(output, 1, replacement=False)
            # output = torch.argmax(output, dim=1)
            hl_choose = torch.squeeze(output).item()
            # if last_hl != -1 and last_hl == hl_choose:
            #     hl_choose = random.randint(0,19)
            # if i <= 120 and i % 10 == 0:
            #     hl_choose = wonderful_seq[(i//20)]
            print(hl_choose)
            # last_hl = hl_choose
            # print(f"state = {state[useful_idx]}")
            # print(f"target = {codes[hl_choose][useful_idx]}")
        #hl_choose = hl_chooses[min(len(hl_chooses)-1,now_state//10)]
        obs = np.concatenate((np.array([state]), np.array([codes[hl_choose][useful_idx]])), axis=1)
        step = model._act(obs)
        action = torch.squeeze(step.action).detach()
        state, reward, done, info = env.step(action)
        # print(list(np.array(state)[np.array([11,12])]))
        # print(info['obs_dict']['goal'])
        ep_r += reward
        # if ep_r == 3 and reward == 1:
        #     print(f"now_state = {now_state}")
        # now_state += 1
        # if now_state == 50:
        #     if ep_r < 1:
        #         now_state = 0
        # if now_state == 80:
        #     if ep_r < 2:
        #         now_state = 60
        # if now_state == 190:
        #     if ep_r < 3:
        #         print("back")
        #         now_state = 150
        # if i % 5 == 0:
        #     img = env.render(mode="rgb_array")
        #     plt.imshow(img)
        #     plt.savefig(f"img/video_{i//5}.jpg")
        #     idx += 1
    print(f"reward = {ep_r}")
    # if ep_r == 2:
    #     print(u)
    #     print(hl_choose)
    #     img = env.render(mode="rgb_array")
    #     plt.imshow(img)
    #     plt.savefig(f"img/fail_3.jpg")
    #     break
    all_reward.append(ep_r)
# x = [i for i in range(100)]
# avg = np.mean(all_reward)
# print(avg) 
# plt.plot(x,all_reward)
# plt.savefig("img/rollout3.png")
# env = gym.make("kitchen-mixed-v0")
# env.set_state(qpos=code[:30],qvel=env.sim.model.key_qvel[0])
# img = env.render(mode="rgb_array")
# plt.imshow(img)
# plt.savefig(f"img/goal.jpg")


