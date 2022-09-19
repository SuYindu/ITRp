from typing import List, Union
import torch
from torch import nn
import torchvision
from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedTokenizer, PreTrainedModel
from data.dataset import MyDataPoint, MyPair


def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.embedding is None:
            return False
    return True


class MyModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        tokenizer: PreTrainedTokenizer,
        text_encoder: PreTrainedModel,
        image_encoder: nn.Module,
        n_classes: int = 2,
        dropout: float = 0,
        text_embedding_length: int = 768,
        image_embedding_length: int = 2048
    ):
        super(MyModel, self).__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.classifier = nn.Linear(text_embedding_length + image_embedding_length, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.device = device
        self.to(device)

    @classmethod
    def from_pretrain(cls, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = 'models'

        # initialize text encoder
        text_encoder_path = f'{models_path}/transformers/{args.text_encoder}'
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
        text_encoder = AutoModel.from_pretrained(text_encoder_path)

        text_embedding_length = {
            'bert-base-uncased': 768,
            'bert-large-uncased': 1024,
            'roberta-base': 768,
        }[args.text_encoder]

        if args.freeze_text:
            for parameter in text_encoder.parameters():
                parameter.requires_grad = False

        # initialize image encoder
        image_encoder = getattr(torchvision.models, args.image_encoder)()
        image_encoder.load_state_dict(torch.load(f'{models_path}/cnn/{args.image_encoder}.pth'))
        if 'efficientnet' in args.image_encoder:
            image_encoder.classifier = torch.nn.Identity()
        else:  # GoogLeNet / ResNet
            image_encoder.fc = torch.nn.Identity()

        image_embedding_length = {
            'resnet101': 2048,
            'resnet152': 2048,
            'efficientnet_b4': 1792,
        }[args.image_encoder]

        if args.freeze_image:
            for parameter in image_encoder.parameters():
                parameter.requires_grad = False

        return cls(
            device, tokenizer, text_encoder, image_encoder,
            args.k, args.dropout,
            text_embedding_length, image_embedding_length
        )

    def _embed_texts(self, pairs: List[MyPair], batch_size: int = 4):
        sentences = [pair.sentence for pair in pairs]
        if use_cache(self.text_encoder, sentences): return

        num_batch = (len(sentences) + batch_size - 1) // batch_size
        batched_sentences = [sentences[i*batch_size:(i+1)*batch_size] for i in range(num_batch)]
        for sentences in batched_sentences:
            texts = [sentence.text for sentence in sentences]
            inputs = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
            output = self.text_encoder(**inputs, return_dict=True)

            lengths = torch.sum(inputs['attention_mask'], dim=1)
            for sentence, length, hidden_state in zip(sentences, lengths, output.last_hidden_state):
                sentence.embedding = torch.mean(hidden_state[:length], dim=0)

    def _embed_images(self, pairs: List[MyPair], batch_size: int = 4):
        images = [pair.image for pair in pairs]
        if use_cache(self.image_encoder, images): return

        num_batch = (len(images) + batch_size - 1) // batch_size
        batched_images = [images[i*batch_size:(i+1)*batch_size] for i in range(num_batch)]
        for images in batched_images:
            embeddings = torch.stack([image.data for image in images]).to(self.device)
            embeddings = self.image_encoder(embeddings)

            for image, embedding in zip(images, embeddings):
                image.embedding = embedding

    def encode(self, pairs: List[MyPair]):
        self._embed_texts(pairs)
        self._embed_images(pairs)
        embeddings_t = torch.stack([pair.sentence.embedding for pair in pairs])
        embeddings_v = torch.stack([pair.image.embedding for pair in pairs])
        embeddings = torch.cat((embeddings_t, embeddings_v), dim=1)
        for pair, embedding in zip(pairs, embeddings):
            pair.embedding = embedding
        return embeddings

    def forward(self, pairs: Union[List[MyPair], MyPair]):
        if isinstance(pairs, MyPair):
            pairs = [pairs]
        embeddings = self.encode(pairs)
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)
        return logits

    def predict(self, data_loader):
        pred_flags = []
        self.eval()
        with torch.no_grad():
            for pairs in data_loader:
                logits = self.forward(pairs)
                pred_flags += torch.argmax(logits, dim=1).tolist()
        return pred_flags
