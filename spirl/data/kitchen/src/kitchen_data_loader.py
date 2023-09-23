import d4rl
import gym
import numpy as np
import itertools
import pickle

from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict

useful_idx = np.array([0,1,2,3,4,5,6,7,8,11,12,17,18,22,23,24,25,26,27,28,29])

class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len:
                continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                                                   for x in self.seqs[fi[0]: fi[1]+1])) for fi in self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        seq = self._sample_seq()
        goal_idx = np.random.randint(self.subseq_len - 1, seq.states.shape[0])
        goal = seq.states[goal_idx][useful_idx]
        goals = np.tile(goal, (self.subseq_len - 1, 1))
        output = AttrDict(
            states=seq.states[goal_idx - self.subseq_len + 1: goal_idx],
            actions=seq.actions[goal_idx - self.subseq_len + 1: goal_idx],
            goals=goals,
            pad_mask=np.ones((self.subseq_len - 1,)),
        )
        # if self.remove_goal:
        #     output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)


class D4RLSequenceSplitDataset_prior(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        with open(self.spec.codebook_dir, 'rb') as f:
            self.codebook = pickle.load(f)
        self.raw_codebook = self.codebook
        with open("spirl/codebook/data.pickle",'rb') as f:
            self.use_data = pickle.load(f)
        self.codebook_norm = np.linalg.norm(self.codebook, axis=-1, keepdims=True) # (20, 12)
        self.codebook = (self.codebook/self.codebook_norm).transpose() # (12, 20)
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len:
                continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                                                   for x in self.seqs[fi[0]: fi[1]+1])) for fi in self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        seq = self._sample_seq()
        goal_idx = np.random.randint(self.subseq_len - 1, seq.states.shape[0])
        goal = seq.states[goal_idx][useful_idx]
        goal = np.expand_dims(goal, axis=0)
        goal = goal/np.linalg.norm(goal,axis=-1,keepdims=True)
        sim_metrix = np.squeeze(np.dot(goal, self.codebook))
        actions = np.argmax(sim_metrix, axis=0)
        expand_actions = [actions for _ in range(self.subseq_len - 1)]
        output = AttrDict(
            states=seq.states[goal_idx - self.subseq_len + 1: goal_idx],
            gt=np.array(expand_actions),
            pad_mask=np.ones((self.subseq_len - 1,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)

def get_codebook():
    with open('spirl/codebook/codebook_50.pickle', 'rb') as f:
        codebook = pickle.load(f) # (20, 60)
    return codebook
