import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import BatchNorm1d


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden, num_classes):
        super(GCN, self).__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        for i in range(n_layers-1):
            in_channels = in_features if i == 0 else hidden
            self.layers.append(GraphConvolution(in_channels, hidden))
        self.last_layer = GraphConvolution(hidden, num_classes)

    def forward(self, x, adj):
        for l in self.layers:
            x = F.relu(l(x, adj))
        x = self.last_layer(x, adj)
        x = F.relu(x)
        # for l in self.layers:
        #     x = (l(x, adj))
        # x = self.last_layer(x, adj)

        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, residual=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        adj = adj + torch.eye(adj.size(-1), device=adj.device)
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adj, support)
        if self.residual:
            output = output + features
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, dropout=0., bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#         self.dropout = dropout
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         torch.nn.init.kaiming_uniform_(self.weight)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, ops, adj):
#         ops = F.dropout(ops, self.dropout, self.training)
#         support = F.linear(ops, self.weight)
#         output = F.relu(torch.matmul(adj, support))
#
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


######################################################################
# GIN
######################################################################
class GIN(nn.Module):
    def __init__(self, n_layers, in_features, hidden, num_classes, num_mlp_layers=2):
        super(GIN, self).__init__()
        self.num_layers = n_layers
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.hidden_dim = hidden
        self.latent_dim = num_classes
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, in_features, hidden, hidden))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden, hidden, hidden))
            self.batch_norms.append(nn.BatchNorm1d(hidden))
        # convert to latent space
        # self.fc = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, node_num = adj.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj, x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        # x = self.fc(x)
        return x

    def forward(self, ops, adj):
        x = self._encoder(ops, adj)
        return x


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


######################################################################
# GIN
######################################################################
# class GIN(torch.nn.Module):
#
#     def __init__(self, n_layers, in_features, hidden, num_classes):
#         # n_layers, in_features, hidden, num_classes
#         super(GIN, self).__init__()
#
#         self.dropout = 0
#         # self.embeddings_dim = [config['hidden_units'][0]] + config['hidden_units']
#         # self.no_layers = len(self.embeddings_dim)
#         self.n_layers = n_layers
#         self.first_h = []
#         self.nns = []
#         self.convs = []
#         self.linears = []
#
#         # train_eps = config['train_eps']
#         # if config['aggregation'] == 'sum':
#         #     self.pooling = global_add_pool
#         # elif config['aggregation'] == 'mean':
#         self.pooling = global_mean_pool
#
#         # for layer, out_emb_dim in enumerate(self.embeddings_dim):
#         for i in range(n_layers):
#
#             if i == 0:
#                 self.first_h = Sequential(Linear(in_features, hidden), BatchNorm1d(hidden), ReLU(),
#                                     Linear(hidden, hidden), BatchNorm1d(hidden), ReLU())
#                 self.linears.append(Linear(hidden, hidden))
#             else:
#                 self.nns.append(Sequential(Linear(hidden, hidden), BatchNorm1d(hidden), ReLU(),
#                                       Linear(hidden, hidden), BatchNorm1d(hidden), ReLU()))
#                 self.convs.append(GINConv(self.nns[-1], train_eps=True))  # Eq. 4.2
#
#                 self.linears.append(Linear(hidden, num_classes))
#
#         self.nns = torch.nn.ModuleList(self.nns)
#         self.convs = torch.nn.ModuleList(self.convs)
#         self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input
#
#     def forward(self, features, adj):
#         # x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = features
#         edge_index = adj.nonzero().t().contiguous()
#
#         out = 0
#
#         for layer in range(self.n_layers):
#             if layer == 0:
#                 x = self.first_h(x)
#                 out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
#             else:
#                 # Layer l ("convolution" layer)
#                 x = self.convs[layer-1](x, edge_index)
#                 out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)
#
#         return out
