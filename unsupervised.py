from pathlib import Path
from tqdm import tqdm
from numpy import argmax
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset

from data.loader import load_dataset_100k, load_dataset_bb
from utils import UniformSampler, get_parser, seed_everything
from utils import compute_features, cluster, train, evaluate_bin, evaluate_quad, save_results


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    train_set = load_dataset_100k(Path('datasets/twitter100k'), args.size, args.shuffle)
    dev_set, test_set = load_dataset_bb(Path('datasets/relationship'))
    test_loader = DataLoader(test_set, batch_size=args.bs * 2, collate_fn=list)

    num_iter = args.size // args.subsize
    indices_list = [list(range(i * args.subsize, (i + 1) * args.subsize)) for i in range(num_iter)]

    model = None

    optimizer_text_encoder = getattr(torch.optim, args.optim_t)(model.text_encoder.parameters(), args.lr_t)
    optimizer_image_encoder = getattr(torch.optim, args.optim_v)(model.image_encoder.parameters(), args.lr_v)

    evaluate = evaluate_bin if args.k == 2 else evaluate_quad

    results = {}
    flags_list_test = []
    f1s_list_test = []
    for epoch in range(args.epochs):
        for i, indices in enumerate(indices_list):
            subset = Subset(train_set, indices)
            train_loader = DataLoader(subset, batch_size=args.bs, collate_fn=list)

            # compute features
            compute_features(tqdm(train_loader), model)

            # cluster and assign pseudo-labels
            centroids = cluster(subset, args.seed, args.k, args.cluster)

            # reset last fully connected layer
            model.classifier.reset_parameters()
            optimizer_c = getattr(torch.optim, args.optim_c)(model.classifier.parameters(), args.lr_c)

            # sampling
            sampler = RandomSampler(subset) if args.sampling == 'random' else UniformSampler(subset, args.sampling)
            train_loader_uniform = DataLoader(subset, args.bs, collate_fn=list, sampler=sampler)

            # train model with pseudo-labels
            optimizers = [optimizer_text_encoder, optimizer_image_encoder, optimizer_c]
            if args.freeze_text:
                optimizers.remove(optimizer_text_encoder)
            if args.freeze_image:
                optimizers.remove(optimizer_image_encoder)
            loss = train(tqdm(train_loader_uniform), model, F.cross_entropy, optimizers)

            # evaluate on three tasks
            flags_test, f1s_test = evaluate(model, tqdm(test_loader))
            flags_list_test.append(flags_test)
            f1s_list_test.append(f1s_test)

            iteration = epoch * num_iter + i + 1
            print(f'iteration #{iteration:02d} epoch #{epoch + 1:02d}, loss: {loss:.3f}')
            print('f1 score: {:2.1%} | {:2.1%}'.format(f1s_test['text'], f1s_test['image']))

        if args.freeze_text and args.freeze_image:
            setting = 'freeze_both'
        elif args.freeze_text:
            setting = 'freeze_text'
        elif args.freeze_image:
            setting = 'freeze_image'
        else:
            setting = 'freeze_none'
        model_id = f"{setting}/{args.text_encoder}+{args.image_encoder}_{args.cluster}_{args.sampling}_" \
                   f"{args.subsize // 1000}k_{args.size // 1000}k{'_shuffle' if args.shuffle else ''}_" \
                   f"seed{args.seed}"

        results = {
            'f1s_text_test': [f1s['text'] for f1s in f1s_list_test],
            'f1s_image_test': [f1s['image'] for f1s in f1s_list_test],
            'flags_text_test': [flags['text'] for flags in flags_list_test],
            'flags_image_test': [flags['image'] for flags in flags_list_test],
        }

        if args.save:
            save_results(f'log/{model_id}.json', results)


    for task in ('text', 'image'):
        print()
        print(f'f1 scores of {task} task')
        print('--------------------------------------------------')
        print('checkpoint   test')
        print('--------------------------------------------------')
        f1s_test = results[f'f1s_{task}_test']
        for epoch, f1_test in zip(range(1, args.epochs * num_iter + 1), f1s_test):
            print(f'{epoch:02d}\t{f1_test:2.1%}')
        print('--------------------------------------------------')


if __name__ == '__main__':
    main()
