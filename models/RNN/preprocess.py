import re
import numpy as np
import pandas as pd
from typing import *
from collections import Counter
DATASET = pd.DataFrame


def clean_text(
    text: str,
) -> None:
    """
    Eliminate unnecessary words using re package
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text
        
        
def train_test_split(
    dataset: DATASET,
    p: float = 0.8,
) -> Tuple[DATASET, DATASET]:
    len_df = len(dataset)
    criterion = np.array(
            [0] * int(len_df * p) + [1] * (len_df - int(len_df * p))
    )
    np.random.shuffle(criterion)
    criterion = criterion.astype(bool)

    train_set = dataset[~criterion]
    test_set = dataset[criterion]

    return train_set, test_set


def sort_and_to_index(
    dataset: DATASET,
    size: int = 20000,
) -> Dict[str, int]:
    """
    Used to sort frequently used words in dataframe and return them to index
    """
    counter = Counter(
         token for tokens in dataset["tokens"] for token in tokens
    )
    sorted_vocab = sorted(
         counter.items(),
         key=(lambda x: x[1]),
         reverse=True,
    )[:size]
    word_to_index = {
         word: index + 2 for index, (word, _) in enumerate(sorted_vocab)
    }
    word_to_index["<PAD>"] = 0
    word_to_index["<OOV>"] = 1

    return word_to_index


def pad_sequence(
    seq: List[int],
    max_len: int = 200,
) -> List[int]:
    if len(seq) < max_len:
        return seq + ([0] * (max_len - len(seq)))

    else:
        return seq[:max_len]
