import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class CNN(nn.Module):
    """CNN model built from pretrained model(resnet 152)"""
    def __init__(self, output_dim=512):
        """
        :param output_dim: output dimension of cnn
        :return: cnn output, encoder of image
        """
        super(CNN, self).__init__()
        # pretrained model resnet152
        pretrained = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(pretrained.children())[:-1])
        self.linear = nn.Linear(pretrained.fc.in_features, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim, momentum=0.01)
        self.init()

    def init(self):
        self.linear.weight.data.normal_(0,0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.resnet(x)
        x = Variable(x.data)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class RNN(torch.nn.Module):
    """
    RNN used for image captioning encoder
    """

    def __init__(self, em_size, hid_size, vocab_size, num_layers=1):
        """
        :param em_size: word embeddings size
        :param hid_size: size of hidden state of recurrent unit
        :param vocab_size: output size
        :param num_layers: number of recurrent layers (default=1)
        """
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, em_size)
        self.rec_unit = nn.LSTM(em_size, hid_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hid_size, vocab_size)

    def forward(self, features, captions, lengths):
        """
        :param features: feature from cnn
        :param captions: target captions
        :param lengths: lengths of image captions
        :returns: prediction
        """
        embedding = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embedding), 1)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        hiddens, _ = self.rec_unit(packed_inputs)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, max_len=25):
        """
        :param features: features from cnn output
        :returns: predicted image captions
        """
        output_ids = []
        states = None
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.rec_unit(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            output_ids.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
        output_ids = torch.stack(output_ids, 1)
        return output_ids.squeeze()