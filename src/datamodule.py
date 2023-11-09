import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class SampleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_data = datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=ToTensor()
        )
        self.valid_data = datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=ToTensor()
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)
