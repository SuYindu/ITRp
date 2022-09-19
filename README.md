# Image-Text Relation Classification in Tweets

## Environment
* python
* numpy
* Pillow
* scikit-learn
* torch
* torchvision
* tqdm
* transformers
* flair (for LSTM+GoogLeNet baseline)
**(pip install -r requirements.txt)**

## Data
Please refer [this repository](https://github.com/danielpreotiuc/text-image-relationship) for the text-image relationship dataset and [this repository](https://github.com/huyt16/Twitter100k) for the Twitter100k dataset.

[controversial_samples.txt](controversial_samples.txt) contains the ids of samples with controversial labels and is used in statistic_relabel.py. Each line in this file contains a head and a list of ids, e.g. "01->11: 3997, 4067, 4299". The head part represents the labels before and after relabeling. For example, "01" stands for "text is not represented & image adds", which is the original label. While "11" stands for "text is represented & image adds", which is the replaced label corrected by us.

## Pretrained Models/Embeddings
Download pretrained BERT-Base from [here](https://huggingface.co/bert-base-uncased/tree/main) and put it in [this directory](resources/transformers).

Download pretrained ResNet-101 from [here](https://download.pytorch.org/models/resnet152-394f9c45.pth), rename the binary file as "resnet101.pth" and put it in [this directory](resources/cnn).

Download pretrained Twitter Word Embedding from [here](https://flair.informatik.hu-berlin.de/resources/embeddings/token/twitter.gensim.vectors.npy) and put it in [this directory](resources/embeddings).

## Usage
### Training & Testing
* run clustering.py for clustering-based baselines.
* run supervised.py for supervised baselines.
* run unsupervised.py for our ITRp method.

### Analysis
* run statistic.py to obtain average F1 score of different tasks on the raw/removed/relabeled test set.
