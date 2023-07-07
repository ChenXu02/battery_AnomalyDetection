import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        feat_path2: str,
        batch_size: int = 64,
        seq_len: int = 120,
        pre_len: int = 1,
        split_ratio: float = 0.8,
        normalize: bool = True,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._feat_path2 = feat_path2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat,self._label,self._ID,self._CandT = utils.data.functions.load_features(self._feat_path)
        self._feat2, self._label2, self._ID2, self._CandT2 = utils.data.functions.load_features(self._feat_path2)
        self._feat_max_val = np.max(self._feat)
        self._feat_max_val_label = np.max(self._label)
        self._feat_max_val2 = np.max(self._feat2)
        self._feat_max_val_label2 = np.max(self._label2)
        print(self._feat_max_val,self._feat_max_val_label,'max')
        print(self._feat_max_val2, self._feat_max_val_label2, 'max')



    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--seq_len", type=int, default=120)
        parser.add_argument("--pre_len", type=int, default=1)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self._label,
            self._ID,
            self._CandT,
            self._feat2,
            self._label2,
            self._ID2,
            self._CandT2,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset))

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def feat_max_val_label(self):
        return self._feat_max_val_label

    @property
    def feat_max_val2(self):
        return self._feat_max_val2

    @property
    def feat_max_val_label2(self):
        return self._feat_max_val_label2
    @property
    def inputdim(self):
        return 17 #dim of V+I
