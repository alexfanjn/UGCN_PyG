import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from utils import *
from sklearn.neighbors._unsupervised import NearestNeighbors




class UGCN(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, number_knn_neighbor, num_layers, num_heads):
        super(UGCN, self).__init__()

        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])

        sparse_adj = edge_index_to_sparse_tensor_adj(data.edge_index)


        full_sparse_adj2 = torch.sparse.mm(sparse_adj, sparse_adj)

        edge_index_2_hop = full_sparse_adj2._indices()[:, torch.where(full_sparse_adj2._values()>=2)[0]]

        edge_index_2_hop, _ = add_self_loops(edge_index_2_hop, num_nodes=data.x.shape[0])


        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(data.x)

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        edge_index_knn = edge_index_knn.long()


        self.data = data
        self.edge_index_2_hop = edge_index_2_hop
        self.edge_index_knn = edge_index_knn


        self.gat1 = GATConv(num_features, num_hidden, heads=num_heads, concat=False)
        self.gat2 = GATConv(num_hidden, num_hidden, heads=num_heads, concat=False)


        self.gat3 = GATConv(num_features, num_hidden, heads=num_heads, concat=False)
        self.gat4 = GATConv(num_hidden, num_hidden, heads=num_heads, concat=False)


        self.gat5 = GATConv(num_features, num_hidden, heads=num_heads, concat=False)
        self.gat6 = GATConv(num_hidden, num_hidden, heads=num_heads, concat=False)

        self.lin1 = torch.nn.Linear(num_hidden, num_classes)
        self.lin2 = torch.nn.Linear(num_hidden, num_classes)
        self.lin3 = torch.nn.Linear(num_hidden, num_classes)
        self.discriminative_agg = torch.nn.Linear(num_classes, 1, bias=False)

        self.dropout = dropout

    def forward(self):
        h1 = self.gat1(self.data.x, self.data.edge_index)
        h1 = F.relu(F.dropout(h1, self.dropout, training=self.training))
        h1 = self.gat2(h1, self.data.edge_index)

        h2 = self.gat3(self.data.x, self.edge_index_2_hop)
        h2 = F.relu(F.dropout(h2, self.dropout, training=self.training))
        h2 = self.gat4(h2, self.edge_index_2_hop)

        h3 = self.gat5(self.data.x, self.edge_index_knn)
        h3 = F.relu(F.dropout(h3, self.dropout, training=self.training))
        h3 = self.gat6(h3, self.edge_index_knn)

        h1_a = self.discriminative_agg(F.tanh(self.lin1(h1)))
        h2_a = self.discriminative_agg(F.tanh(self.lin2(h2)))
        h3_a = self.discriminative_agg(F.tanh(self.lin3(h3)))

        h_all_a = torch.cat([h1_a, h2_a, h3_a], dim=1)


        h_all_a = torch.softmax(h_all_a, dim=1)


        h = h_all_a[:, 0].reshape(-1,1) * h1 + h_all_a[:, 1].reshape(-1,1) * h2 + h_all_a[:, 2].reshape(-1,1) * h3

        return F.log_softmax(h, 1)

