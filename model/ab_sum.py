# Building Encoder seq2seq model with RNN and Transformer for abstractive summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # for decoder, we use n_directions = 1
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        # the output of the decoder is a word in the target language, so the dimension of the output is the vocabulary size of the target language
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, input_dim, output_dim, teacher_forcing_ratio):
        batch_size = input_dim.shape[1]
        target_len = output_dim.shape[0]
        target_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        hidden, cell = self.encoder(input_dim)
        input = output_dim[0,:]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (output_dim[t] if teacher_force else top1)
        return outputs
    
class Seq2Seq_trainer(object):
    def __init__(self, model, train_iterator, valid_iterator, pad_index, device, clip, lr):
        # initialize the model
        self.model = model.to(device)
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.clip = clip
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad_index)
        self.model.apply(self.init_weights)
        print(f'The model has {self.count_parameters(self.model):,} trainable parameters')

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_iterator):
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()
            output = self.model(src, trg, 0.5)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_iterator)
    
# Test the model
if __name__ == "__main__":
    input_dim = 10
    output_dim = 10
    encoder = Encoder(input_dim, 256, 512, 2, 0.5)
    decoder = Decoder(output_dim, 256, 512, 2, 0.5)
    model = Seq2Seq(encoder, decoder, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(model)