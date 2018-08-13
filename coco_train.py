from coco_utils import *
from coco_models import *
import argparse
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from coco_vocab import load_vocabulary

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_path = 'data/train2014'
train_caption_path = 'data/annotations/captions_train2014.json'
val_path = 'data/val2014'
val_caption_path = 'data/annotations/captions_val2014.json'


class COCOTrain:
    def __init__(self, arg):
        self.epoch = 5
        self.batch_size = arg.batch_size
        self.checkpoint = arg.checkpoint_file
        self.log_step = arg.log_step
        self.save_dir = arg.save_dir
        self.embed_size = arg.embed_size
        self.num_hidden = arg.num_hidden
        self.sv_step = 500
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-4
        self.train_losses = []
        self.vocab = load_vocabulary()
        self.val_losses = []
        self.encoder = CNN(self.embed_size)
        self.decoder = RNN(self.embed_size, self.num_hidden, len(self.vocab), 1)
        self.train_loader = get_coco_loader(path=train_path,
                                            file=train_caption_path,
                                            vocab=self.vocab,
                                            transform=data_transform,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=1)
        self.val_loader = get_coco_loader(path=val_path,
                                          file=val_caption_path,
                                          vocab=self.vocab,
                                          transform=data_transform,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=1)
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.current_step = 0

    def initialize(self):
        if self.checkpoint is not None:
            encoder_state_dict, decoder_state_dict, self.optimizer, *meta = load_model(self.checkpoint)
            self.current_epoch, _, losses_train, losses_val = meta
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
        else:
            params = list(self.decoder.parameters()) + \
                     list(self.encoder.linear.parameters()) + list(self.encoder.batchnorm.parameters())
            self.optimizer = optim.Adam(params, lr=self.lr)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def train(self):
        self.initialize()
        try:
            for epoch in range(self.current_epoch, self.current_epoch + self.epoch):
                print("Epoch: ", epoch)
                self.scheduler.step()
                for step, (images, captions, lengths) in enumerate(self.train_loader, start=0):
                    # print("Step: ", step)
                    # Set mini-batch dataset
                    # print("Set mini-batch image")
                    self.current_step = step
                    images = to_variable(images, volatile=True)
                    # print("Set mini-batch caption")
                    captions = to_variable(captions)
                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                    # print("Forward")
                    # Forward, Backward and Optimize
                    self.decoder.zero_grad()
                    self.encoder.zero_grad()

                    # run on single GPU
                    features = self.encoder(images)
                    outputs = self.decoder(features, captions, lengths)
                    # print("Loss")
                    train_loss = self.criterion(outputs, targets)
                    self.train_losses.append(train_loss.data[0])
                    # print("Updating")
                    print("Step: %5d, loss: %8.4f" % (step, train_loss.data[0]))
                    train_loss.backward()
                    self.optimizer.step()

                    # Save the models
                    if (step + 1) % self.sv_step == 0:
                        save_model(self.encoder, self.decoder, self.optimizer, epoch, step,
                                   self.train_losses, self.save_dir)
                        save_losses(self.train_losses, os.path.join(self.save_dir, 'losses.pkl'))
                self.current_epoch += 1

        except KeyboardInterrupt:
            pass
        finally:
            # Do final save
            save_model(self.encoder, self.decoder, self.optimizer, self.current_epoch,
                       self.current_step, self.train_losses, self.save_dir)
            save_losses(self.train_losses, os.path.join(self.save_dir, 'losses.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str, default=None, help='path to saved checkpoint')
    parser.add_argument('--save_dir', type=str, default='model-checkpoints', help='directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches')
    parser.add_argument('--log_step', type=int, default=250, help='number of steps in between calculating loss')
    parser.add_argument('--num_hidden', type=int, default=512, help='number of hidden units in the RNN')
    parser.add_argument('--embed_size', type=int, default=512, help='number of embeddings in the RNN')
    args = parser.parse_args()
    coco = COCOTrain(args)
    coco.train()
