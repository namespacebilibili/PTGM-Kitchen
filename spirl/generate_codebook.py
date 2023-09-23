from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import os, imageio, sys, pickle
from scipy.spatial.distance import euclidean
import argparse
from tqdm import tqdm
import pickle

useful_idx = np.array([0,1,2,3,4,5,6,7,8,11,12,17,18,22,23,24,25,26,27,28,29])

def plot_embedding(data, y, save_path, centers=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    plt.scatter(data[:,0], data[:,1], c=y)

    if centers is not None:
        centers = (centers - x_min) / (x_max - x_min)
        plt.scatter(centers[:,0], centers[:,1], c='r')
    plt.savefig('codebook/tsne_class10_robot.png')

def sample_embeddings(data, interval=6, extract_dim=21):
    
    state_dim = 60
    codes = np.empty(shape=(0,state_dim))
    embeds = np.empty(shape=(0,extract_dim))
    states = data["states"]
    length = states.shape[0]
    sample_idx = np.arange(8, length, interval)
    for idx in sample_idx:
        cat_state = np.expand_dims(states[idx], axis=0)
        codes = np.concatenate((codes, cat_state))
        cat_state = states[idx][useful_idx]
        cat_state = np.expand_dims(cat_state, axis=0)
        embeds = np.concatenate((embeds, cat_state))

    print(f"codes = {codes.shape}")
    print(f"embeds = {embeds.shape}")
    return sample_idx, codes, embeds


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    exp_name = '{}_n{}_s{}'.format(args.embedding, args.n_codebook, args.seed)
    output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data_dir,'rb') as f:
        data = pickle.load(f)

    sample_idx, codes, embeds = sample_embeddings(data,extract_dim=args.extract_dim)

    if args.embedding == 'tsne':
        tsne = TSNE(n_components=2)
        result = tsne.fit_transform(embeds)
    else:
        raise NotImplementedError

    print('running kmeans clustering')
    kmeans = KMeans(n_clusters=args.n_codebook, init='k-means++', random_state=args.seed).fit(result)
    y_pred = kmeans.labels_
    centers = kmeans.cluster_centers_
    center_idxs = []
    print(f"y_pred = {y_pred}, centers = {centers}")
    print('done')

    for iclust in range(kmeans.n_clusters):
        cluster_pts = result[kmeans.labels_ == iclust]
        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
        cluster_cen = centers[iclust]
        min_idx = np.argmin([euclidean(result[idx], cluster_cen) for idx in cluster_pts_indices])
        idx = cluster_pts_indices[min_idx]
        center_idxs.append(idx)

    with open(f"codebook/codebook_robot{args.n_codebook}.pickle", 'wb') as f:
        pickle.dump(np.array(embeds[center_idxs]),f)
    with open(f"codebook/codebook_robot{args.n_codebook}_full.pickle", 'wb') as f:
        pickle.dump(np.array(codes[center_idxs]),f)
    print(embeds[center_idxs])
    plot_embedding(result, y_pred, output_dir, result[center_idxs])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='tsne')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='./codebook')
    parser.add_argument('--data-dir', type=str, default='./codebook/data.pickle')
    parser.add_argument('--n-codebook', type=int, default=20)
    parser.add_argument('--extract_dim', type=int, default=20)

    args = parser.parse_args()
    main(args)

    # with open("./codebook/codebook.pickle", 'rb') as f:
    #     codebook = pickle.load(f)

    # print(codebook)
