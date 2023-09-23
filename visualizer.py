from spirl.rl.envs.kitchen import KitchenEnv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from spirl.utils.general_utils import AttrDict

env_config = AttrDict(
    reward_norm=1.,
)

with open("/home/yhq/Desktop/spirl-master/spirl/codebook/codebook.pickle", 'rb') as f:
    codebook = pickle.load(f) # (100, 60)

idx = 1
tmp = None
env = KitchenEnv(env_config)
state = np.random.random((1,60))
img = env.render(mode='rgb_array')
for code in codebook:
    env.reset(code = code)
    img = env.render(mode='rgb_array')
    print(img)
    plt.imshow(img)
    plt.savefig(f"image/code_{idx}.png")
    idx += 1
    print(code == tmp)
    tmp = code