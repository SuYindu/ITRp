import re
import csv
import random
from pathlib import Path
from data.dataset import MySentence, MyImage, MyPair, MyDataset


def normalize_tweet(tweet: str):
    url_re = r' http[s]?://t.co/\w+$'
    tweet = re.sub(url_re, '', tweet)
    return tweet


def load_dataset_100k(path: Path, size: int = 100000, shuffle: bool = False):
    assert path.exists()

    with open(path/'text.txt', encoding='utf-8') as txt_file:
        pairs = [MyPair(MySentence(text), MyImage(f'{i+1}.jpg')) for i, text in enumerate(txt_file)]
    if shuffle:
        random.seed(0)
        random.shuffle(pairs)

    return MyDataset(pairs[:size], path/'images')


def load_dataset_bb(path: Path, split: int = 3576, normalize: bool = False):
    assert path.exists()

    with open(path/'data.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
        pairs = [MyPair(sentence=MySentence(normalize_tweet(row['tweet']) if normalize else row['tweet']),
                        image=MyImage(f"T{row['tweet_id']}.jpg"),
                        text_flag=int(row['text_is_represented']),
                        image_flag=int(row['image_adds']))
                 for row in csv_reader]

    return MyDataset(pairs[:split], path/'images'), MyDataset(pairs[split:], path/'images')


if __name__ == '__main__':
    train_set = load_dataset_100k(Path('../datasets/twitter100k'))
    dev_set, test_set = load_dataset_bb(Path('../datasets/relationship'))
