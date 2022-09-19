import timeit
from pathlib import Path
from tqdm import tqdm
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from data.loader import load_dataset_100k, load_dataset_bb
from utils import get_parser, seed_everything, init_model, compute_features, save_results

split = 3576

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    train_set = load_dataset_100k(Path('datasets/twitter100k'), args.size)
    dev_set, test_set = load_dataset_bb(Path('datasets/relationship'))
    train_loader = DataLoader(train_set, batch_size=args.bs, collate_fn=list)
    test_loader = DataLoader(test_set, batch_size=args.bs, collate_fn=list)

    device = torch.device(f'cuda:{args.cuda}')
    model = init_model(args, device)

    # compute features
    compute_features(tqdm(train_loader), model)
    x_train = torch.stack([pair.embedding for pair in train_set]).cpu().numpy()
    compute_features(tqdm(test_loader), model)
    x_dev_test = torch.stack([pair.embedding for pair in test_set]).cpu().numpy()

    start_time = timeit.default_timer()

    if args.cluster == 'random':
        from random import randint
        pred_flags = [randint(0, 3) for _ in range(len(test_set))]
    else:
        if args.cluster == 'kmeans':
            clustering = KMeans(4, random_state=args.seed).fit(x_train)
            pred_flags = clustering.predict(x_dev_test)
        elif args.cluster == 'gaussian':
            clustering = GaussianMixture(4, random_state=args.seed, covariance_type='tied').fit(x_train)
            pred_flags = clustering.predict(x_dev_test)
        else:
            clustering = None

    end_time = timeit.default_timer()

    true_flags_tv = [pair.text_image_flag for pair in test_set]
    best_accuracy, best_map = 0, None
    for flag_map in permutations(range(4)):
        mapped_flags = [flag_map[pred_flag] for pred_flag in pred_flags]
        accuracy = accuracy_score(true_flags_tv, mapped_flags)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_map = flag_map

    mapped_flags = [best_map[pred_flag] for pred_flag in pred_flags]
    f1_tv_test = f1_score(true_flags_tv, mapped_flags, average='weighted')

    text_map = [0, 0, 1, 1]
    true_flags_t = [pair.text_flag for pair in test_set]
    mapped_flags_t = [text_map[mapped_flag] for mapped_flag in mapped_flags]
    f1_t_test = f1_score(true_flags_t, mapped_flags_t, average='weighted')

    image_map = [0, 1, 0, 1]
    true_flags_v = [pair.image_flag for pair in test_set]
    mapped_flags_v = [image_map[mapped_flag] for mapped_flag in mapped_flags]
    f1_v_test = f1_score(true_flags_v, mapped_flags_v, average='weighted')

    print('clustering takes {:2.2f}s'.format(end_time - start_time))
    print('f1 on test set: {:2.1%} | {:2.1%} | {:2.1%}'.format(f1_t_test, f1_v_test, f1_tv_test))

    if args.save:
        results = {
            'f1s_text_test': [f1_t_test],
            'f1s_image_test': [f1_v_test],
            'f1s_text_image_test': [f1_tv_test],
            'flags_text_test': [mapped_flags_t],
            'flags_image_test': [mapped_flags_v],
            'flags_text_image_test': [mapped_flags],
        }
        model_id = f'{args.text_encoder}+{args.image_encoder}_{args.cluster}_{args.size // 1000}k_{args.seed}'
        save_results(f'log/clustering/{model_id}', results)


if __name__ == '__main__':
    main()
