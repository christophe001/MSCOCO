import torch
import pickle
from torch.autograd import Variable
import os
from pycocotools.coco import COCO
import torch.utils.data as data
from PIL import Image
import nltk
import re
from itertools import takewhile


def to_variable(x, volatile=False):
    """
    transform to torch variable

    :param x:
    :param volatile:
    :return:
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def save_model(encoder, decoder, optimizer, epoch, step, train_losses, path):
    """
    function to save currently trained model

    :param encoder: encoder of model(cnn)
    :param decoder: decoder of model(rnn)
    :param optimizer: optimizer used for training
    :param epoch: train epoch
    :param step: train step
    :param train_losses: model train loss
    :param path: path to save model
    :return: none
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, 'model-%d-%d.ckpt' % (epoch + 1, step + 1))
    print('Saving model to:', file)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer': optimizer,
        'epoch': epoch,
        'step': step,
        'train_losses': train_losses}, file)


def load_model(file_path):
    """
    function to load existing model

    :param file_path: model file path
    :return: saved model params
    """
    model = torch.load(file_path)
    encoder_state_dict = model['encoder_state_dict']
    decoder_state_dict = model['decoder_state_dict']
    optimizer = model['optimizer']
    epoch = model['epoch']
    step = model['step']
    train_losses = model['train_losses']
    return encoder_state_dict, decoder_state_dict, optimizer, epoch, step, train_losses


def save_losses(train_losses, val_losses, path):
    """
    save losses

    :param train_losses: train losses
    :param val_losses: validation losses(optional)
    :param path: path to save loss
    :return:
    """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path, 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f, protocol=2)


def translation(idx, vocab):
    blacklist = [vocab.word2idx[word] for word in [vocab.start_token()]]
    predict = lambda word_id: vocab.idx2word[word_id] != vocab.end_token()
    sampled_caption = [vocab.idx2word[word_id] for word_id in takewhile(predict, idx) if word_id not in blacklist]
    sentence = ' '.join(sampled_caption)
    return sentence


class CocoDataset(data.Dataset):
    def __init__(self, path, file, vocab=None, transform=None):
        """
        :param path: images directory path
        :param file: annotations file path
        :param vocab: vocabulary
        :param transform: torchvision.transforms
        """
        self.path = path
        self.coco = COCO(file)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        :param index: annotation id
        return: (image, caption)
        """
        coco = self.coco
        vocab = self.vocab
        annotation_id = self.ids[index]
        caption = coco.anns[annotation_id]['caption']
        image_id = coco.anns[annotation_id]['image_id']
        path = coco.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.path, path))
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption_str = str(caption).lower()
        tokens = nltk.tokenize.word_tokenize(caption_str)
        caption = torch.Tensor([vocab(vocab.start_token())] +
                               [vocab(token) for token in tokens] +
                               [vocab(vocab.end_token())])

        return image, caption


def collate_fun(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    caption_lengths = [len(caption) for caption in captions]
    padded_captions = torch.zeros(len(captions), max(caption_lengths)).long()
    for ix, caption in enumerate(captions):
        end = caption_lengths[ix]
        padded_captions[ix, :end] = caption[:end]
    return images, padded_captions, caption_lengths


def get_coco_loader(path, file, vocab, transform=None,
                         batch_size=32, shuffle=True, num_workers=2):
    """Returns custom COCO Dataloader"""
    coco = CocoDataset(path=path,
                       file=file,
                       vocab=vocab,
                       transform=transform)

    data_loader = data.DataLoader(dataset=coco,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fun)

    return data_loader


class ImageDataset(data.Dataset):
    def __init__(self, dir_path, transform=None):
        """
        :param dir_path: image path
        :param transform: image preprocessing schedule
        :returns:
        """
        self.dir_path = dir_path
        self.transform = transform

        _a, _b, files = next(os.walk(dir_path))
        self.file_names = files

    def __len__(self):
        """
        :returns:
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        :param idx:
        :returns:
        """
        # load image and caption
        file_name = self.file_names[idx]
        image_id = re.findall('[0-9]{12}', file_name)[0]
        image_path = os.path.join(self.dir_path, file_name)
        image = Image.open(image_path).convert('RGB')

        # transform image
        if self.transform is not None:
            image = self.transform(image)

        return image, image_id


def get_image_loader(dir_path, transform, batch_size=32, shuffle=True, num_workers=2):
    """
    Returns torch.utils.data.DataLoader for custom coco dataset.
    :param dir_path:
    :param ann_path:
    :param vocab:
    :param transform:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :returns:
    """
    image_data = ImageDataset(dir_path=dir_path, transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = data.DataLoader(dataset=image_data, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    return data_loader
