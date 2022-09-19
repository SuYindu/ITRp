from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.loader import load_dataset_bb
from model.model import MyModel
from model.model_lstm import MyLstmModel
from utils import get_parser, seed_everything, train, evaluate, save_results


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.epochs, args.optim, args.lr = 10, 'Adam', 2e-5
    print(args)

    seed_everything(args.seed)
    Model = MyLstmModel if 'lstm' in args.text_encoder else MyModel

    train_set, test_set = load_dataset_bb(Path('datasets/relationship'))
    train_loader = DataLoader(train_set, batch_size=args.bs, collate_fn=list, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.bs * 2, collate_fn=list)

    flags_task, f1s_task = {}, {}
    for task in ('text', 'image', 'text_image'):
        flags_task[task], f1s_task[task] = [], []

        args.k = 4 if task == 'text_image' else 2
        model = Model.from_pretrain(args)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

        print(f'******************** {task} task ********************')
        best_epoch, best_f1 = 0, 0
        for epoch in range(1, args.epochs + 1):
            loss = train(tqdm(train_loader), model, F.cross_entropy, optimizer, task=task)
            pred_flags, f1_weighted = evaluate(model, tqdm(test_loader), task)
            flags_task[task].append(pred_flags)
            f1s_task[task].append(f1_weighted)
            if f1_weighted > best_f1:
                best_epoch, best_f1 = epoch, f1_weighted

            print(f'epoch #{epoch:02d}, loss: {loss:.3f}, test f1 score: {f1_weighted:2.1%}')

        print(f'best f1 score ({best_f1:2.1%}) at epoch#{best_epoch}')

    if args.save:
        results = {
            'f1s_text_test': f1s_task['text'],
            'f1s_image_test': f1s_task['image'],
            'f1s_text_image_test': f1s_task['text_image'],
            'flags_text_test': flags_task['text'],
            'flags_image_test': flags_task['image'],
            'flags_text_image_test': flags_task['text_image'],
        }
        model_id = f'{args.text_encoder}+{args.image_encoder}_{args.seed}'
        save_results(f'log/supervised/{model_id}', results)


if __name__ == '__main__':
    main()
