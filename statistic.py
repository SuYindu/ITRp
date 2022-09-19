import json
import csv
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score


def weighted_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


examples_dict = {}
with open('controversial_samples.txt', 'r') as txt_file:
    lines = txt_file.readlines()
    for line in lines:
        head, content = line.strip().split(': ')
        indices = [int(index) for index in content.split(', ')]
        examples_dict[head] = indices
selected_t = examples_dict['00->10'] + examples_dict['01->11'] + examples_dict['10->00'] + examples_dict['11->01']
selected_v = examples_dict['00->01'] + examples_dict['01->00'] + examples_dict['10->11'] + examples_dict['11->10']
selected_tv = examples_dict['00->11'] + examples_dict['01->10'] + examples_dict['10->01'] + examples_dict['11->00']


split = 3576
set_path = Path('datasets/relationship')
with open(set_path / 'data.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
    true_flags_t, true_flags_v = [], []
    for row in csv_reader:
        true_flags_t.append(int(row['text_is_represented']))
        true_flags_v.append(int(row['image_adds']))
    repl_flags_t, repl_flags_v = true_flags_t[:], true_flags_v[:]
    for i in selected_t + selected_tv:
        repl_flags_t[i] = 1 - repl_flags_t[i]
    for i in selected_v + selected_tv:
        repl_flags_v[i] = 1 - repl_flags_v[i]
    true_flags = [text_flag * 2 + image_flag for text_flag, image_flag in zip(true_flags_t, true_flags_v)]
    repl_flags = [text_flag * 2 + image_flag for text_flag, image_flag in zip(repl_flags_t, repl_flags_v)]
    true_flags_t, true_flags_v, true_flags = true_flags_t[split:], true_flags_v[split:], true_flags[split:]
    repl_flags_t, repl_flags_v, repl_flags = repl_flags_t[split:], repl_flags_v[split:], repl_flags[split:]


parser = argparse.ArgumentParser()
parser.add_argument('--text_filter', default='bert-base-uncased+resnet101', type=str)
parser.add_argument('--image_filter', default='bert-base-uncased+resnet101', type=str)
parser.add_argument('--number', default=6, type=int)
parser.add_argument('--epoch', default=0, type=int)
args = parser.parse_args()


pred_flags_dict = {}
log_path_dict = {'text': Path(f'log/supervised'), 'image': Path(f'log/supervised')}
for task in ('text', 'image'):
    pred_flags_dict[task] = []
    files = sorted([file for file in log_path_dict[task].iterdir()])
    for file in files:
        if getattr(args, f'{task}_filter') not in str(file):
            continue

        # print(f'reading {file}')
        with open(file, 'r') as json_file:
            content = json.load(json_file)
            flags = content[f'flags_{task}_test']
            f1s = [weighted_f1_score(repl_flags_t if task == 'text' else repl_flags_v, pred_flag) for pred_flag in flags]
            epoch = np.argmax(f1s) if args.epoch == 0 else args.epoch - 1
            pred_flags_dict[task].append(flags[epoch])

    # if len(pred_flags_dict[task]) > args.number:
    #     indices = np.argsort(best_f1s_task[task])[-args.number:].tolist()
    #     pred_flags_dict[task] = [pred_flags_dict[task][i] for i in indices]
    assert len(pred_flags_dict[task]) == args.number
    print(f'{len(pred_flags_dict[task])} candidates for {task} task')

pred_flags_tv = [[text_flag * 2 + image_flag for text_flag, image_flag in zip(text_flags, image_flags)]
                 for text_flags, image_flags in zip(pred_flags_dict['text'], pred_flags_dict['image'])]


selected = selected_tv + selected_t + selected_v
filtered_indices = list(filter(lambda idx: idx + split not in selected, range(len(true_flags))))
true_flags_filtered_t = [true_flags_t[i] for i in filtered_indices]
true_flags_filtered_v = [true_flags_v[i] for i in filtered_indices]
true_flags_filtered = [true_flags[i] for i in filtered_indices]
pred_flags_filtered_t = [[flags[i] for i in filtered_indices] for flags in pred_flags_dict['text']]
pred_flags_filtered_v = [[flags[i] for i in filtered_indices] for flags in pred_flags_dict['image']]
pred_flags_filtered = [[flags[i] for i in filtered_indices] for flags in pred_flags_tv]

f1s_raw_t = [weighted_f1_score(true_flags_t, pred_flags) for pred_flags in pred_flags_dict['text']]
f1s_raw_v = [weighted_f1_score(true_flags_v, pred_flags) for pred_flags in pred_flags_dict['image']]
f1s_raw = [weighted_f1_score(true_flags, pred_flags) for pred_flags in pred_flags_tv]

f1s_filtered_t = [weighted_f1_score(true_flags_filtered_t, pred_flag_t) for pred_flag_t in pred_flags_filtered_t]
f1s_filtered_v = [weighted_f1_score(true_flags_filtered_v, pred_flag_v) for pred_flag_v in pred_flags_filtered_v]
f1s_filtered = [weighted_f1_score(true_flags_filtered, pred_flag_tv) for pred_flag_tv in pred_flags_filtered]

f1s_replaced_t = [weighted_f1_score(repl_flags_t, pred_flag_t) for pred_flag_t in pred_flags_dict['text']]
f1s_replaced_v = [weighted_f1_score(repl_flags_v, pred_flag_v) for pred_flag_v in pred_flags_dict['image']]
f1s_replaced = [weighted_f1_score(repl_flags, pred_flags) for pred_flags in pred_flags_tv]

for heading, (f1s_t, f1s_v, f1s_tv) in zip(('raw', 'removed', 'replaced'), (
        (f1s_raw_t, f1s_raw_v, f1s_raw),
        (f1s_filtered_t, f1s_filtered_v, f1s_filtered),
        (f1s_replaced_t, f1s_replaced_v, f1s_replaced)
)):
    print('------------------------------')
    print(heading)
    print(f'{np.mean(f1s_t)*100:2.1f}\t{np.mean(f1s_v)*100:2.1f}\t{np.mean(f1s_tv)*100:2.1f}')
    # for f1_t, f1_v, f1 in zip(f1s_t, f1s_v, f1s):
    #     print(f'{f1_t*100:2.1f} | {f1_v*100:2.1f} | {f1*100:2.1f}')
    # print()
print('------------------------------')
print()
