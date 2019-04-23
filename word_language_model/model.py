import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        if rnn_type in ('LSTM','GRU'):
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            self.rnn = nn.RNN(ninp, nhid, nlayers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken,ninp)
        self.liner = nn.Linear(nhid,ntoken)
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nlayers = nlayers
        self.nhid = nhid
    def forward(self,input,hidden):
        input = self.embed(input)
        input = self.dropout(input)
        output,hidden = self.rnn(input,hidden)
        output = self.dropout(output)
        reoutput = self.liner(output.view(output.size(0)*output.size(1),output.size(2)))
        return reoutput.view(output.size(0),output.size(1),reoutput.size(1)),hidden

    def init_hidden(self,bsz):
        weight = list(self.parameters())[0]
        if self.rnn_type =='LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
