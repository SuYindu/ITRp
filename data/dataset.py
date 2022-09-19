from typing import Optional, List
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataPoint:
    def __init__(self):
        self.embedding: Optional[Tensor] = None


class MySentence(MyDataPoint):
    def __init__(self, text: str):
        super().__init__()
        self.text: str = text


class MyImage(MyDataPoint):
    def __init__(self, file_name: str):
        super().__init__()
        self.file_name: str = file_name
        self.data: Optional[Tensor] = None


class MyPair(MyDataPoint):
    def __init__(self, sentence: MySentence, image: MyImage,
                 text_flag: int = -1, image_flag: int = -1, pseudo_flag: int = -1):
        super().__init__()
        self.sentence: MySentence = sentence
        self.image: MyImage = image
        self.text_flag: int = text_flag
        self.image_flag: int = image_flag
        self.pseudo_flag: int = pseudo_flag
        self.text_image_flag: int = text_flag * 2 + image_flag


class MyDataset(Dataset):
    def __init__(self, pairs: List[MyPair], images_path: Path):
        self.pairs: List[MyPair] = pairs
        self.images_path: Path = images_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = pair.image
        if image.data is None:
            path_to_image = self.images_path/image.file_name
            image.data = Image.open(path_to_image).convert('RGB')
            image.data = self.transform(image.data)
        return pair


class MyCorpus:
    def __init__(self, train: Optional[MyDataset], dev: Optional[MyDataset], test: Optional[MyDataset]):
        self.train: MyDataset = train
        self.dev: MyDataset = dev
        self.test: MyDataset = test
