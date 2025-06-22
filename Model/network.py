from .models import Cheb_GCN, graph_learning
from .layers import SwiGLU
import torch
from torch import nn
import torch.nn.functional as F


class _Global_Message_Model(nn.Module):
    def __init__(self, DATASET_Dict, Hidden_size=32, word_emb=32, num_heads=4):
        super(_Global_Message_Model, self).__init__()

        self.Modal_Index = DATASET_Dict['Modal_Index']
        self.Embedding_dict = nn.ModuleDict()

        self.Hidden_size = Hidden_size
        self.num_heads = num_heads
        self.word_emb = word_emb
        self.seq_len = len(self.Modal_Index)

        self.modal_embedding_list = []
        for index_, modal_index in enumerate(self.Modal_Index):
            self.Embedding_dict[f'Modal_{index_}'] = nn.Linear(len(modal_index), self.word_emb)
        
        self.Fusion_modal_0 = nn.MultiheadAttention(self.word_emb, self.num_heads, batch_first=True)
        self.layer_norm_0 = nn.LayerNorm(self.word_emb)
        self.Fusion_modal_1 = nn.MultiheadAttention(self.word_emb, self.num_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(self.word_emb)


        self.sequeeze_layer = nn.Sequential(nn.Linear(self.word_emb*self.seq_len, self.word_emb*(self.seq_len//2)),
                                            nn.ReLU(),
                                            nn.Linear(self.word_emb*(self.seq_len//2), self.Hidden_size),
                                            nn.ReLU(),
                                            )


    def forward(self, X):
        
        self.modal_embedding_list = []
        for index_, modal_index in enumerate(self.Modal_Index):
            self.modal_embedding_list.append(self.Embedding_dict[f'Modal_{index_}'](X[:, modal_index]))
        
        Modal_embedding = torch.stack(self.modal_embedding_list, 1)
        Y = self.message(Modal_embedding)

        return Y

    def message(self, Modal_embedding):
        
        Modal_embedding_0, _ = self.Fusion_modal_0(Modal_embedding, Modal_embedding, Modal_embedding)
        Modal_embedding_0 = self.layer_norm_0(Modal_embedding + Modal_embedding_0)

        Modal_embedding_1, _ = self.Fusion_modal_1(Modal_embedding_0, Modal_embedding_0, Modal_embedding_0)
        Modal_embedding = self.layer_norm_0(Modal_embedding_0 + Modal_embedding_1)

        Y = self.sequeeze_layer(Modal_embedding.reshape(-1, self.word_emb*self.seq_len))

        return Y
    

class _Edge_message_Model(nn.Module):
    def __init__(self, U_index, V_index, Hidden_size=16, word_emb=4, num_heads=4):
        super(_Edge_message_Model, self).__init__()

        self.U_channels_Index = U_index
        self.V_channels_Index = V_index

        self.Hidden_size = Hidden_size
        self.num_heads = num_heads
        self.word_emb = word_emb
        self.seq_len = self.Hidden_size // self.word_emb

        self.Linear_U = nn.Linear(len(self.U_channels_Index), self.Hidden_size)
        self.Linear_V = nn.Linear(len(self.V_channels_Index), self.Hidden_size)

        self.Modal_Fusion = nn.MultiheadAttention(self.word_emb, self.num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.word_emb)

        self.mask = torch.full((self.seq_len*2, self.seq_len*2), float('-inf'))
        self.mask[:self.seq_len, :self.seq_len] = 0
        self.mask[self.seq_len:, self.seq_len:] = 0

        self.sequeeze_layer = nn.Sequential(nn.Linear(self.Hidden_size*2, self.Hidden_size, bias=False),
                                            nn.ReLU(),)

    def forward(self, X):
        
        U = F.relu(self.Linear_U(X[:, self.U_channels_Index]))
        V = F.relu(self.Linear_V(X[:, self.V_channels_Index]))

        Y = self.message(U, V)

        return Y

    def message(self, U, V):
        U = U.view(-1, self.seq_len, self.word_emb)
        V = V.view(-1, self.seq_len, self.word_emb)

        Q = torch.cat([U, V], dim=1)
        KV = torch.cat([V, U], dim=1)

        Y, _ = self.Modal_Fusion(Q, KV, KV, attn_mask=self.mask.to(self.Linear_U.weight.device))
        Y = self.layer_norm(Q + Y)

        Y = self.sequeeze_layer(Y.reshape(-1, self.Hidden_size*2))

        return Y

class _View_graph_Model(nn.Module):
    def __init__(self, Herter_Graph_Lable_View, Hidden_size):
        super(_View_graph_Model, self).__init__()

        self.Herter_Graph_Lable_View = Herter_Graph_Lable_View
        self.Feature_Index_dict, self.Edge_list = self.Herter_Graph_Lable_View
        self._Edge_num = len(self.Edge_list)
        self._Edge_embedding = []

        self._Edge_model = nn.ModuleDict()
        for Edge_index, (u, v) in enumerate(self.Edge_list):
            self._Edge_model[f'Edge_model_{Edge_index}'] = _Edge_message_Model(self.Feature_Index_dict[u], self.Feature_Index_dict[v], Hidden_size)

        self.swiglu = SwiGLU(self._Edge_num)

    # def forward(self, X):
    #
    #     self._Edge_embedding = []
    #     for Edge_index in range(self._Edge_num):
    #
    #         Edge_embedding = self._Edge_model[f'Edge_model_{Edge_index}'](X)
    #         self._Edge_embedding.append(Edge_embedding)
    #
    #     Y = self.message(self._Edge_embedding)
    #
    #     return Y

    def forward(self, X):
        futures = []

        for Edge_index in range(self._Edge_num):
            futures.append(torch.jit.fork(self._Edge_model[f'Edge_model_{Edge_index}'], X))

        self._Edge_embedding = [torch.jit.wait(f) for f in futures]

        Y = self.message(self._Edge_embedding)
        return Y
        
    def message(self, embedding):

        Y = torch.stack(embedding, dim=-1)
        Y = torch.sum(self.swiglu(Y), dim=-1)
        
        return Y


class _Label_graph_Model(nn.Module):
    def __init__(self, Herter_Graph_Lable, Hidden_size):
        super(_Label_graph_Model, self).__init__()

        self.Herter_Graph_Lable = Herter_Graph_Lable
        self._View_num = len(self.Herter_Graph_Lable)
        self._View_embedding = []

        self._View_model = nn.ModuleDict()
        for View_index, Herter_Graph_Lable_View in enumerate(self.Herter_Graph_Lable):
            self._View_model[f'View_model_{View_index}'] = _View_graph_Model(Herter_Graph_Lable_View, Hidden_size)

        self.swiglu = SwiGLU(self._View_num)

    # def forward(self, X):
    #
    #     self._View_embedding = []
    #     for View_index in range(self._View_num):
    #
    #         View_embedding = self._View_model[f'View_model_{View_index}'](X)
    #         self._View_embedding.append(View_embedding)
    #
    #     Y = self.message(self._View_embedding)
    #
    #     return Y

    def forward(self, X):

        futures = []

        for View_index in range(self._View_num):
            futures.append(torch.jit.fork(self._View_model[f'View_model_{View_index}'], X))

        self._View_embedding = [torch.jit.wait(f) for f in futures]

        Y = self.message(self._View_embedding)

        return Y
    
    def message(self, embedding):

        Y = torch.stack(embedding, dim=-1)
        Y = torch.sum(self.swiglu(Y), dim=-1)

        return Y
    

class HeterGraph_Model_Kmeans(nn.Module):
    def __init__(self, DATASET_Dict, Herter_Graph, Hidden_size, Drop_rate, K):
        super(HeterGraph_Model_Kmeans, self).__init__()

        self.Hidden_size = Hidden_size
        self.DATASET_Dict = DATASET_Dict
        self.Herter_Graph = Herter_Graph

        self._Label_num = self.DATASET_Dict['Class_Num']
        self._Label_embedding = []
        self._Auxi_classifier_output = []

        self._Label_model = nn.ModuleDict()
        self._Auxi_classifier = nn.ModuleDict()
        for Label_index, Herter_Graph_Lable in enumerate(self.Herter_Graph):
            self._Label_model[f'Label_model_{Label_index}'] = _Label_graph_Model(Herter_Graph_Lable, Hidden_size)
            self._Auxi_classifier[f'Label_model_{Label_index}'] = nn.Linear(self.Hidden_size, 2)
        
        self.Adj_Learning = graph_learning(self.DATASET_Dict, self.Hidden_size, rate=0.1)
        self.GCN = Cheb_GCN(Dim_emb=Hidden_size, hidden=Hidden_size // 2, out_channels=self.DATASET_Dict['Class_Num'],
                            P=Drop_rate, K=K)
        # self.Classfier = nn.Linear(self.Hidden_size, self._Label_num)

        self.Global_Message = _Global_Message_Model(self.DATASET_Dict, Hidden_size=self.Hidden_size, word_emb=self.Hidden_size)

        self.Message_MLP = nn.Sequential(nn.Linear(self.DATASET_Dict['Class_Num'] * self.Hidden_size, (self.DATASET_Dict['Class_Num'] * self.Hidden_size)//2),
                                         nn.ReLU(),
                                         nn.Linear((self.DATASET_Dict['Class_Num'] * self.Hidden_size)//2, self.Hidden_size),
                                         nn.ReLU(),
                                        )

    # def forward(self, X):
    #
    #     self._Label_embedding = []
    #     self._Auxi_classifier_output = []
    #     for Label_index in  range(self._Label_num):
    #
    #         Label_embedding = self._Label_model[f'Label_model_{Label_index}'](X)
    #         Auxi_classifier_output = self._Auxi_classifier[f'Label_model_{Label_index}'](Label_embedding)
    #
    #         self._Label_embedding.append(Label_embedding)
    #         self._Auxi_classifier_output.append(Auxi_classifier_output)
    #
    #
    #     Y = self.message(self._Label_embedding)
    #
    #     Global_Embedding = self.Global_Message(X)
    #
    #     Adj = self.Adj_Learning(X)
    #     Y = self.GCN(Y + Global_Embedding, Adj)
    #
    #     # Y = self.Classfier(Y + Global_Embedding)
    #
    #     return Y, self._Label_embedding, self._Auxi_classifier_output

    def forward(self, X):
        futures = []

        for Label_index in range(self._Label_num):
            future = torch.jit.fork(self._process_label, Label_index, X)
            futures.append(future)

        global_message_future = torch.jit.fork(self.Global_Message, X)
        adj_learning_future = torch.jit.fork(self.Adj_Learning, X)

        results = [torch.jit.wait(f) for f in futures]
        Global_Embedding = torch.jit.wait(global_message_future)
        Adj = torch.jit.wait(adj_learning_future)

        self._Label_embedding = [res[0] for res in results]
        self._Auxi_classifier_output = [res[1] for res in results]


        Y = self.message(self._Label_embedding)
        Y = self.GCN(Y + Global_Embedding, Adj)

        return Y, self._Label_embedding, self._Auxi_classifier_output


    def _process_label(self, Label_index, X):

        Label_embedding = self._Label_model[f'Label_model_{Label_index}'](X)
        Auxi_classifier_output = self._Auxi_classifier[f'Label_model_{Label_index}'](Label_embedding)

        return Label_embedding, Auxi_classifier_output


    def message(self, embedding):

        # Y = torch.sum(torch.stack(embedding), dim=0)
        Y = self.Message_MLP(torch.cat(embedding, dim=-1))

        return Y