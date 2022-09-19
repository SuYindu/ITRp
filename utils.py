import os
import argparse
import random
import json
from itertools import permutations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data.sampler import Sampler


class UniformSampler(Sampler):
    def __init__(self, dataset, strategy):
        self.dataset = dataset
        self.indices = self.generate_indices(strategy)

    def generate_indices(self, strategy):
        index_lists, res = {}, []
        for i, pair in enumerate(self.dataset):
            if pair.pseudo_flag not in index_lists:
                index_lists[pair.pseudo_flag] = [i]
            else:
                index_lists[pair.pseudo_flag].append(i)

        sizes = [len(index_list) for index_list in index_lists.values()]
        min_size, max_size, mean_size = np.min(sizes), np.max(sizes), int(np.mean(sizes))
        size = min_size if strategy == 'under' else max_size if strategy == 'over' else mean_size
        for index_list in index_lists.values():
            res += np.random.choice(index_list, size, replace=len(index_list) < size).tolist()
        np.random.shuffle(res)

        return res

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2, choices=(2, 4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--size', type=int, default=100000)
    parser.add_argument('--subsize', type=int, default=5000)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--optim_t', type=str, default='Adam', choices=('SGD', 'Adam'))
    parser.add_argument('--lr_t',      type=float, default=2e-5)
    parser.add_argument('--optim_v', type=str, default='SGD', choices=('SGD', 'Adam'))
    parser.add_argument('--lr_v',      type=float, default=3e-3)
    parser.add_argument('--optim_c', type=str, default='Adam', choices=('SGD', 'Adam'))
    parser.add_argument('--lr_c',      type=float, default=1e-4)
    parser.add_argument('--text_encoder', type=str, default='bert-base-uncased',
                        choices=('lstm', 'bert-base-uncased', 'bert-large-uncased', 'roberta-base'))
    parser.add_argument('--image_encoder', type=str, default='resnet101',
                        choices=('googlenet', 'resnet101', 'resnet152', 'efficientnet_b4'))
    parser.add_argument('--cluster', type=str, default='kmeans',
                        choices=('kmeans', 'gaussian', 'random'))
    parser.add_argument('--sampling', type=str, default='combination',
                        choices=('under', 'over', 'combination', 'random'))
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--freeze_text', action='store_true', default=False)
    parser.add_argument('--freeze_image', action='store_true', default=False)
    parser.add_argument('--use_centroid', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False)
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def compute_features(loader, model):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            model.encode(batch)


def cluster(dataset, seed, k=2, algorithm='kmeans'):
    x = torch.stack([pair.embedding for pair in dataset]).cpu().numpy()
    centroids = None

    if algorithm == 'kmeans':
        kmeans = KMeans(k, random_state=seed).fit(x)
        pseudo_flags = kmeans.predict(x)
        centroids = kmeans.cluster_centers_
    elif algorithm == 'gaussian':
        gm = GaussianMixture(k, covariance_type='tied', random_state=seed).fit(x)
        pseudo_flags = gm.predict(x)
    else:
        pseudo_flags = [random.randint(0, k-1) for _ in range(len(dataset))]

    for pair, pseudo_flag in zip(dataset, pseudo_flags):
        pair.pseudo_flag = int(pseudo_flag)

    return centroids


def train(loader, model, criteria, optimizers, task='pseudo'):
    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]

    model.train()

    losses = []
    for batch in loader:
        output = model(batch)
        target = torch.tensor([getattr(sample, f'{task}_flag') for sample in batch]).to(output.device)
        loss = criteria(output, target)
        losses.append(loss.item())
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    return np.mean(losses)


def evaluate(model, loader, task):
    true_flags = [getattr(pair, f'{task}_flag') for batch in loader for pair in batch]
    pred_flags = model.predict(loader)
    return pred_flags, f1_score(true_flags, pred_flags, average='weighted')


def evaluate_quad(model, loader):
    true_flags = [pair.text_image_flag for batch in loader for pair in batch]
    pred_flags = model.predict(loader)

    best_accuracy, best_map = 0, None
    for flag_map in permutations(range(4)):
        mapped_flags = [flag_map[pred_flag] for pred_flag in pred_flags]
        accuracy = accuracy_score(true_flags, mapped_flags)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_map = flag_map

    mapped_flags = [best_map[pred_flag] for pred_flag in pred_flags]
    f1_text_image = f1_score(true_flags, mapped_flags, average='weighted')

    text_map = [0, 0, 1, 1]
    true_flags_text = [pair.text_flag for batch in loader for pair in batch]
    mapped_flags_text = [text_map[mapped_flag] for mapped_flag in mapped_flags]
    f1_text = f1_score(true_flags_text, mapped_flags_text, average='weighted')

    image_map = [0, 1, 0, 1]
    true_flags_image = [pair.image_flag for batch in loader for pair in batch]
    mapped_flags_image = [image_map[mapped_flag] for mapped_flag in mapped_flags]
    f1_image = f1_score(true_flags_image, mapped_flags_image, average='weighted')

    return f1_text, f1_image, f1_text_image


def evaluate_bin(model, loader):
    pred_flags = model.predict(loader)
    flags, f1s = {}, {}

    for task in ('text', 'image'):
        true_flags = [getattr(pair, f'{task}_flag') for batch in loader for pair in batch]

        best_accuracy, best_map = 0, None
        for flag_map in permutations(range(2)):
            mapped_flags = [flag_map[pred_flag] for pred_flag in pred_flags]
            accuracy = accuracy_score(true_flags, mapped_flags)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_map = flag_map

        mapped_flags = [best_map[pred_flag] for pred_flag in pred_flags]
        flags[task] = mapped_flags
        f1s[task] = f1_score(true_flags, mapped_flags, average='weighted')

    return flags, f1s


def save_results(file_name, results):
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)
