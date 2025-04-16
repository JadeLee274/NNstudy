from config import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer


tokenizer = Tokenizer()

loader = DataLoader(
    extension=('.en', '.de'),
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
    init_token='<sos>',
    eos_token='<eos>',   
)

train, valid, test = loader.make_dataset()

loader.build_vocab(
    train_data=train,
    min_frequency=2,
)

train_iter, valid_iter, test_iter = loader.make_iter(
    train=train,
    validate=valid,
    test=test,
    batch_size=batch_size,
    device=device,
)

src_pad_idx = loader.source.vocab.stoi['<pad>']
target_pad_idx = loader.target.vocab.stoi['<pad>']
target_sos_idx = loader.target.vocab.stoi['<sos>']

encoder_vocab_size = len(loader.source.vocab)
decoder_vocab_size = len(loader.target.vocab)
