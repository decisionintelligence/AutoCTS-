
import numpy as np

from .gcn_net import GCN
from ..genotypes import PRIMITIVES
from .set_encoder.setenc_models import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_task_oriented_ahc(data_loader, valid_loader, ahc, criterion, optimizer):
    train_loss = []
    num_correct = 0
    ahc.train()
    train_dataloader = data_loader

    train_dataloader.shuffle()

    for (task_feature, arch0, arch1, label) in train_dataloader.get_iterator():  # 对每个batch
        label = torch.Tensor(label).to(DEVICE)
        outputs = ahc(arch0, arch1, task_feature)
        loss = criterion(outputs, label)
        train_loss.append(loss.item())
        pred = torch.round(outputs)
        num_correct += torch.eq(pred, label).sum().float().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy = num_correct / train_dataloader.size

    # eval
    with torch.no_grad():
        ahc = ahc.eval()
        valid_loss = []
        num_correct = 0
        for i, (task_feature, arch0, arch1, label) in enumerate(valid_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)

            outputs = ahc(arch0, arch1, task_feature)

            pred = torch.round(outputs)
            loss = criterion(outputs, label)
            valid_loss.append(loss.item())
            num_correct += torch.eq(pred, label).sum().float().item()

        accuracy = num_correct / valid_loader.size

    return np.mean(train_loss), train_accuracy, np.mean(valid_loss), accuracy


def evaluate(test_loader, ahc, criterion):
    ahc = ahc.eval()
    with torch.no_grad():
        test_loss = []
        num_correct = 0
        for i, (task_feature, arch0, arch1, label) in enumerate(test_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)

            outputs = ahc(arch0, arch1, task_feature)

            pred = torch.round(outputs)
            loss = criterion(outputs, label)
            test_loss.append(loss.item())
            num_correct += torch.eq(pred, label).sum().float().item()

        accuracy = num_correct / test_loader.size

    return accuracy, np.mean(test_loss)


def geno_to_adj(arch):
    # arch.shape = [7, 2]
    # 输出邻接矩阵，和节点特征
    # 这里的邻接矩阵对应op为顶点的DAG，和Darts相反
    # GCN处理无向图，这里DAG是有向图，所以需要改改？？？参考Wei Wen的文章
    node_num = len(arch) + 2  # 加上一个input和一个output节点
    adj = np.zeros((node_num, node_num))
    ops = [len(PRIMITIVES)]
    for i in range(len(arch)):
        connect, op = arch[i]
        ops.append(arch[i][1])
        if connect == 0 or connect == 1:
            adj[connect][i + 1] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 1] = 1
            adj[(connect - 2) * 2 + 3][i + 1] = 1
    adj[-3][-1] = 1
    adj[-2][-1] = 1  # output
    ops.append(len(PRIMITIVES) + 1)

    return adj, ops


class AHC(nn.Module):
    def __init__(self, n_nodes, n_ops, n_layers=2, ratio=2, embedding_dim=128):
        # 后面要参考下Wei Wen文章的GCN实现
        super(AHC, self).__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops

        # +2用于表示input和output node
        self.embedding = nn.Embedding(self.n_ops + 2, embedding_dim=embedding_dim)
        self.gcn = GCN(n_layers=n_layers, in_features=embedding_dim,
                       hidden=embedding_dim, num_classes=embedding_dim)
        self.nz = 256
        self.fz = 128
        self.intra_setpool = SetPool(dim_input=256,
                                     num_outputs=1,
                                     dim_output=self.nz,
                                     dim_hidden=self.nz,
                                     mode='sabPF')
        self.inter_setpool = SetPool(dim_input=self.nz,
                                     num_outputs=1,
                                     dim_output=self.nz,
                                     dim_hidden=self.nz,
                                     mode='sabP')
        self.set_fc = nn.Sequential(
            nn.Linear(self.nz, self.fz), nn.ReLU())

        self.graph_fc = nn.Sequential(
            nn.Linear(embedding_dim * ratio, self.fz), nn.ReLU())

        self.pred_fc = nn.Sequential(nn.Linear(self.fz * ratio, self.fz, bias=True), nn.ReLU(),
                                     nn.Linear(self.fz, 1, bias=True))

    def forward(self, arch0, arch1, embedding_feature):

        # 先将数组编码改成邻接矩阵编码
        # arch0.shape = [batch_size, 7, 2]
        b_adj0, b_adj1, b_ops0, b_ops1, features = [], [], [], [], []
        for i in range(len(arch0)):
            adj0, ops0 = geno_to_adj(arch0[i])
            adj1, ops1 = geno_to_adj(arch1[i])
            b_adj0.append(adj0)
            b_adj1.append(adj1)
            b_ops0.append(ops0)
            b_ops1.append(ops1)

            feature = torch.Tensor(embedding_feature[i]).to(DEVICE)
            features.append(feature)

        # extract the arch feature
        b_adj0 = np.array(b_adj0)
        b_adj1 = np.array(b_adj1)
        b_ops0 = np.array(b_ops0)
        b_ops1 = np.array(b_ops1)

        b_adj0 = torch.Tensor(b_adj0).to(DEVICE)
        b_adj1 = torch.Tensor(b_adj1).to(DEVICE)
        b_ops0 = torch.LongTensor(b_ops0).to(DEVICE)
        b_ops1 = torch.LongTensor(b_ops1).to(DEVICE)

        embedd1 = self.extract_features((b_adj0, b_ops0))
        embedd2 = self.extract_features((b_adj1, b_ops1))

        # extract the task feature

        task_feature = self.set_encode(features)
        self.task_feature = task_feature

        feature = torch.cat([embedd1, embedd2], dim=1)

        task_logits = self.set_fc(task_feature)
        graph_logits = self.graph_fc(feature)

        logits = self.pred_fc(torch.concat([graph_logits, task_logits], dim=1)).squeeze(1)
        probility = torch.sigmoid(logits)

        return probility

    def set_encode(self, X):
        proto_batch = []
        for x in X:
            cls_protos = self.intra_setpool(x).squeeze(1)
            proto_batch.append(
                cls_protos)

        v = torch.stack(proto_batch)
        v = self.inter_setpool(v).squeeze(1)
        return v

    def extract_features(self, arch):
        # 分别输入邻接矩阵和operation？
        if len(arch) == 2:
            matrix, op = arch
            return self._extract(matrix, op)
        else:
            print('error')

    def _extract(self, matrix, ops):

        ops = self.embedding(ops)
        feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)  # shape=[b, nodes, dim] pooling

        return feature


if __name__ == '__main__':
    arch = [[0, 1], [0, 2], [1, 0], [0, 0], [2, 2], [2, 5], [3, 0]]
    adj, ops = geno_to_adj(arch)
    print(adj)
    print(ops)
