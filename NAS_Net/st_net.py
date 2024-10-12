from .operations import *
from .genotypes import PRIMITIVES


class Cell(nn.Module):

    def __init__(self, genotype, C, steps, supports, nodevec1, nodevec2, cheb=None, alpha=None):
        super(Cell, self).__init__()

        self.preprocess = Identity()
        self._steps = steps
        self.alpha = alpha
        self.cheb = cheb
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.supports = supports

        indices, op_names = zip(*genotype)
        self._compile(C, indices, op_names)

    def _compile(self, C, indices, op_names):
        assert len(op_names) == len(indices)
        print(len(PRIMITIVES))
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):

            if 'diff_gcn' in PRIMITIVES[name]:
                op = OPS[PRIMITIVES[name]](C, self.supports, self.nodevec1, self.nodevec2, True)
            elif 'cheb_gcn' in PRIMITIVES[name]:
                op = OPS[PRIMITIVES[name]](C, self.cheb, self.nodevec1, self.nodevec2, self.alpha)
            else:
                op = OPS[PRIMITIVES[name]](C)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0):
        s0 = self.preprocess(s0)

        states = [s0]
        for i in range(self._steps):
            if i == 0:
                # assert 0 == self._indices[0]
                h1 = states[0]
                op1 = self._ops[0]
                h1 = op1(h1)
                h2 = torch.zeros_like(h1)
            else:
                h1 = states[self._indices[2 * i - 1]]
                h2 = states[self._indices[2 * i]]
                op1 = self._ops[2 * i - 1]
                op2 = self._ops[2 * i]
                h1 = op1(h1)
                h2 = op2(h2)  # 因为有bias导致不为0
            s = h1 + h2
            states += [s]
        return states[-1]


class Network(nn.Module):

    def __init__(self, adj_mx, scaler, args, arch):
        super(Network, self).__init__()
        self._adj_mx = adj_mx
        self._args = args
        self._scaler = scaler
        self._layers = args.layers
        self._steps = args.steps

        # for diff_gcn
        new_adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        supports = [torch.tensor(i).to(DEVICE) for i in new_adj]

        # adaptive adjacency matrix
        # self.alpha = nn.Parameter(torch.randn(len(supports), len(supports)).to(DEVICE), requires_grad=True).to(DEVICE)
        if args.randomadj:
            aptinit = None
        else:
            aptinit = supports[0]
        if aptinit is None:  # 注意，后续计算梯度时，若网络中没有用到gcn算子，则不应该计算这部分，因为grad.data一定为None
            self.nodevec1 = nn.Parameter(torch.randn(args.num_nodes, 10).to(DEVICE), requires_grad=True).to(DEVICE)
            self.nodevec2 = nn.Parameter(torch.randn(10, args.num_nodes).to(DEVICE), requires_grad=True).to(DEVICE)
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(DEVICE)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(DEVICE)

        # input layer
        self.start_linear = linear(args.in_dim, args.hid_dim)
        # self.start_linear = nn.Conv1d(args.in_dim, args.hid_dim, 1)
        # position encoding
        self.pe = PositionalEmbedding(args.hid_dim)

        C = args.hid_dim
        self.cells = nn.ModuleList()
        self.skip_connect = nn.ModuleList()
        for i in range(self._layers):
            cell = Cell(arch, C, self._steps, supports, self.nodevec1, self.nodevec2)
            self.cells += [cell]
            self.skip_connect.append(nn.Conv2d(C, args.hid_dim * 8, (1, 1)))
            # self.skip_connect.append(nn.Conv1d(C, args.hid_dim * 8, 1))
            
        # output layer
        self.end_linear_1 = linear(c_in=args.hid_dim * 8, c_out=args.hid_dim * 16)  # 改成4和256试试？
        if args.task == 'multi':
            self.end_linear_2 = linear(c_in=args.hid_dim * 16, c_out=args.output_len)
        else:
            self.end_linear_2 = linear(c_in=args.hid_dim * 16, c_out=1)

        # self.end_linear_1 = nn.Conv1d(args.hid_dim * 8, args.hid_dim * 16, 1)
        # self.end_linear_2 = nn.Conv1d(args.hid_dim * 16, args.seq_len, 1)
    def forward(self, input):
        # b, c, N, T = input.shape
        # input = input.reshape(b, c, -1)
        # print(f'The input shape:{input.shape}')
        x = self.start_linear(input)
        # x = x.reshape(b, x.shape[1], N, T)

        b, D, N, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, T, D)  # [64, 207, 12, 32]
        x = x * math.sqrt(self._args.hid_dim)
        x = x + self.pe(x)
        x = x.reshape(b, -1, T, D).permute(0, 3, 1, 2)  # 这为什么也影响score???

        skip = 0
        for i, cell in enumerate(self.cells):
            x = cell(x)
            # x = x.reshape(b, D, -1)
            skip = self.skip_connect[i](x) + skip
            # x = x.reshape(b, D, N, T)
        # skip = skip.reshape(b, skip.shape[1], N, T)
        out = F.relu(skip)
        out = torch.max(out, dim=-1, keepdim=True)[0]
        # state = state.reshape(b, skip.shape[1], -1)
        out = self.end_linear_1(out)

        out = F.relu(out)

        out = self.end_linear_2(out)
        # logits = logits.reshape(b, logits.shape[1], N, -1)

        return out
