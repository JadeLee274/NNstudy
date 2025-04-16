from typing import *
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(
        self,
        extension: Tuple[str, str],
        tokenize_en,
        tokenize_de,
        init_token,
        eos_token,
    ) -> None:
        self.extension = extension
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('Dataset Initializing Start')
    
    def make_dataset(self):
        if self.extension == ('.de', '.en'):
            self.source = Field(
                tokenize=self.tokenize_de,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True,
            )
            self.target = Field(
                tokenize=self.tokenize_en,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True,
            )
        elif self.extension == ('.en', '.de'):
            self.source = Field(
                tokenize=self.tokenize_en,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True,
            )
            self.target = Field(
                tokenize=self.tokenize_de,
                init_token=self.init_token,
                eos_token=self.eos_token,
                lower=True,
                batch_first=True,
            )

        train_data, valid_data, test_data = Multi30k.splits(
            exts=self.extension,
            fields=(self.source, self.target),
        )

        return train_data, valid_data, test_data
    
    def build_vocab(
        self,
        train_data,
        min_frequency,
    ) -> None:
        self.source.build_vocab(train_data, min_freq=min_frequency)
        self.target.build_vocab(train_data, min_freq=min_frequency)
    
    def make_iter(
        self,
        train,
        validate,
        test,
        batch_size: int,
        device,
    ):
        train_iterator, val_iterator, test_iterator = BucketIterator.splits(
            datasets=(train, validate, test),
            batch_size=batch_size,
            device=device,
        )
        print('Dataset Initializing Done')
        return train_iterator, val_iterator, test_iterator