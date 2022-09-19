from typing import List, Union
import torch
from torch import nn
import torchvision
from transformers import AutoModel
import flair
from flair.embeddings import TokenEmbeddings
from data.dataset import MyDataPoint, MyPair


def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.embedding is None:
            return False
    return True


class MyLstmModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        token_embedding: nn.Module,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        n_classes: int = 2,
        dropout: float = 0,
        text_embedding_length: int = 768,
        image_embedding_length: int = 2048
    ):
        super(MyLstmModel, self).__init__()
        self.token_embedding = token_embedding
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.classifier = nn.Linear(text_embedding_length + image_embedding_length, n_classes)
        self.dropout = nn.Dropout(dropout)

        flair.device = device
        self.device = device
        self.to(device)

    @classmethod
    def from_pretrain(cls, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = 'models'

        # initialize text encoder
        assert args.text_encoder == 'lstm'
        flair.cache_root = 'models'
        token_embedding = flair.embeddings.WordEmbeddings('twitter')
        text_encoder = nn.LSTM(token_embedding.embedding_length, hidden_size=200, batch_first=True)
        text_embedding_length = 200

        if args.freeze_text:
            for parameter in text_encoder.parameters():
                parameter.requires_grad = False

        # initialize image encoder
        if 'googlenet' in args.image_encoder:
            kwargs = {"transform_input": True, "aux_logits": False, "init_weights": False}
            original_aux_logits = kwargs["aux_logits"]
            kwargs["aux_logits"] = True
            model = torchvision.models.GoogLeNet(**kwargs)
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
            image_encoder = model
        else:
            image_encoder = getattr(torchvision.models, args.image_encoder)()
        image_encoder.load_state_dict(torch.load(f'{models_path}/cnn/{args.image_encoder}.pth'))
        if 'efficientnet' in args.image_encoder:
            image_encoder.classifier = torch.nn.Identity()
        else:  # GoogLeNet / ResNet
            image_encoder.fc = torch.nn.Identity()

        image_embedding_length = {
            'googlenet': 1024,
            'resnet101': 2048,
            'resnet152': 2048,
            'efficientnet_b4': 1792,
        }[args.image_encoder]

        if args.freeze_image:
            for parameter in image_encoder.parameters():
                parameter.requires_grad = False

        return cls(
            device, token_embedding, text_encoder, image_encoder,
            args.k, args.dropout,
            text_embedding_length, image_embedding_length
        )

    def _embed_texts(self, pairs: List[MyPair]):
        sentences = [pair.sentence for pair in pairs]
        flair_sentences = [flair.data.Sentence(sentence.text) for sentence in sentences]
        batch_size = len(flair_sentences)
        lengths = [len(flair_sentence) for flair_sentence in flair_sentences]
        max_length = max(lengths)
        embedding_length = self.token_embedding.embedding_length
        zero_tensor = torch.zeros(embedding_length * max_length, dtype=torch.float, device=self.device)

        self.token_embedding.embed(flair_sentences)
        embedding_list = []
        for flair_sentence in flair_sentences:
            embedding_list += [token.embedding for token in flair_sentence]
            num_padding = max_length - len(flair_sentence)
            if num_padding > 0:
                padding = zero_tensor[:embedding_length * num_padding]
                embedding_list.append(padding)
        embeddings = torch.cat(embedding_list).view(batch_size, max_length, embedding_length)

        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.text_encoder(embeddings)
        embeddings = h.squeeze(0)

        for sentence, embedding in zip(sentences, embeddings):
            sentence.embedding = embedding

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
