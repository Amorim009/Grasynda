import torch
import torch.nn as nn
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
      for name,param in m.named_parameters():
        if 'weight_ih' in name:
          init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
          init.orthogonal_(param.data)
        elif 'bias' in name:
          param.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    def __init__(self, opt):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.z_dim, num_layers=opt.num_layer, batch_first=True)
        self.fc = nn.Linear(opt.z_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat
