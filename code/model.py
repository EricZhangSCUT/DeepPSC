import torch
import torch.nn as nn
from .resnet import resnet50

class LinearLN(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.Linear = nn.Linear(input_dim, output_dim)
        self.LayerNorm = nn.LayerNorm(output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.Linear(x)
        x = self.LayerNorm(x)
        x = self.activation(x)
        return x


class ConvLocalStruEmbedBlocks(nn.Module):
    def __init__(self, output_dim, input_dim=5):
        super().__init__()
        self.resnet = resnet50(output_dim=output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class LSTMGlobalizingBlocks(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = int(output_dim//2)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self._ln = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, lengths):
        x_batch = []
        last = 0
        for length in lengths:
            next_ = last + length
            x_batch.append(x[last: next_].view(length, -1))
            last = next_

        x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        x_batch = torch.nn.utils.rnn.pack_padded_sequence(
            x_batch, lengths, batch_first=True)
        x_batch = self.lstm(x_batch)[0]
        x_batch, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x_batch, batch_first=True)

        x_batch = self.activation(self._ln(x_batch))
        x_batch = [x_batch[i, :lengths[i]] for i in range(len(lengths))]
        return x_batch


class JointPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.buffering_layer = LinearLN(2*input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_batch):
        x_batch = [torch.cat((x[:-1], x[1:]), 1) for x in x_batch]
        x = torch.cat(x_batch, 0)
        x = self.buffering_layer(x)
        x = self.output(x)
        return x


class DeepPSC(nn.Module):
    def __init__(self, dims=[5, 512, 1024, 4]):
        super().__init__()

        input_dim, local_dim, global_dim, output_dim = dims

        self.local_structure_embeding_block = ConvLocalStruEmbedBlocks(
            input_dim=input_dim, output_dim=local_dim).cuda()
        self.local_structure_embeding_blocks = nn.DataParallel(
            self.local_structure_embeding_block)

        self.local_structure_feature_globalizing_blocks = LSTMGlobalizingBlocks(
            input_dim=local_dim, output_dim=global_dim).cuda()

        self.predicting_blocks = JointPredictor(
            input_dim=global_dim, output_dim=output_dim).cuda()

    def forward(self, x, lens=None):
        x = self.local_structure_embeding_blocks(x, lens)
        x = self.local_structure_feature_globalizing_blocks(x, lens)
        x = self.predicting_blocks(x)
        return x

    def save_model(self, save_name):
        checkpoint = {}
        checkpoint['local'] = self.local_structure_embeding_block.state_dict()
        checkpoint['global'] = self.local_structure_feature_globalizing_blocks.state_dict()
        checkpoint['predict'] = self.predicting_blocks.state_dict()
        torch.save(checkpoint, '%s.pth.tar' % save_name)

    def load_model(self, train_name, model_name, blocks=['local', 'global', 'predict']):
        checkpoint = torch.load(
            '../output/%s/model/%s_model.pth.tar' % (train_name, model_name))

        self.local_structure_embeding_block.load_state_dict(
            checkpoint['local'])
        self.local_structure_embeding_blocks = nn.DataParallel(
            self.local_structure_embeding_block)

        self.local_structure_feature_globalizing_blocks.load_state_dict(
            checkpoint['global'])
        self.predicting_blocks.load_state_dict(checkpoint['predict'])
