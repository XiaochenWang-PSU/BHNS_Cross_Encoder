from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn, optim
import torch.functional as F
from sentence_transformers import SentenceTransformer
from torch.nn.functional import softmax, binary_cross_entropy_with_logits, cross_entropy
from torch.nn import CosineSimilarity
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import nn
    

class Theta_Estimation(nn.Module):
    def __init__(self):
        super().__init__()
        # Define sentence transformer
        self.st = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    def forward(self, q, k, label):
        emb_q = self.st(q)["sentence_embedding"]
        emb_k = self.st(k)["sentence_embedding"]
        norm_q = emb_q / emb_q.norm(dim=1)[:, None]
        q_q_sim = torch.mm(norm_q, norm_q.transpose(0, 1))
        theta = torch.zeros((emb_q.shape[0], emb_q.shape[0]))

        # Determine unique documents and group queries by related document
        unique_docs, inverse_indices = torch.unique(emb_k, return_inverse=True, dim=0)
        for doc_idx in range(len(unique_docs)):
            # Find queries related to the current document
            related_queries_indices = (inverse_indices == doc_idx).nonzero(as_tuple=True)[0]
            T_j = len(related_queries_indices)

            if T_j > 0:
                for i in range(emb_q.shape[0]):
                    sum_r_j = label[related_queries_indices].sum()
                    theta[i, related_queries_indices] = (label[related_queries_indices] * q_q_sim[i, related_queries_indices]).sum(dim=-1) / sum_r_j

        return theta  








    
class Efficient_Hard_negative_generator(nn.Module):    
    def __init__(self):
        super().__init__()

#         define sentence transformer
        
        self.st = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        
        
        
    def forward(self, q, k, label, num_neg, tau):


        
        emb_q = self.st(q)["sentence_embedding"]
        emb_k = self.st(k)["sentence_embedding"]

        batch_size = emb_q.shape[0]
        
        q_d_sim = torch.diag(label)

        q_q_sim = torch.matmul(emb_q, emb_q.t())
        q_q_sim = torch.matmul(emb_q, emb_q.t())/ ((torch.sqrt((emb_q * emb_q).sum(-1)))*(torch.sqrt((emb_q * emb_q).sum(-1))))
        
        output = q_d_sim * (1-q_q_sim)**tau 
        top_k = torch.topk(q_d_sim, num_neg,  dim=-1).indices

        return top_k
    
    
    

    
    

class Efficient_Pseudo_labeler(nn.Module):    
    def __init__(self):
        super().__init__()

#         define sentence transformer
        
        self.st = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        

        
    def forward(self, q, labels, indices, neg_chunk_size):


        
        emb_q = self.st(q)['sentence_embedding']
        batch_size = emb_q.shape[0]
        
        # extend the embedding of query such that it can be used for torch.gather
        emb_q_3d = emb_q.unsqueeze(0).repeat(batch_size, 1, 1)
        labels = labels.unsqueeze(1).unsqueeze(1).repeat(1, batch_size, 1)
        indices_l = indices.unsqueeze(-1).to(emb_q_3d.device)
        indices_q = indices.unsqueeze(-1).repeat(1, 1, 384).to(emb_q_3d.device)
        emb_q_ = torch.gather(emb_q_3d, 1, indices_q).reshape(neg_chunk_size * batch_size, -1)
        labels = torch.gather(labels, 1, indices_l).reshape(neg_chunk_size * batch_size, -1)
        
        
        
        
        emb_q = emb_q.unsqueeze(1).repeat(1, neg_chunk_size, 1).reshape(neg_chunk_size * batch_size, -1)

        
        # calcualte cos sim between queries
        weight = (emb_q * emb_q_).sum(-1)
        weight = weight / ((torch.sqrt((emb_q * emb_q).sum(-1)))*(torch.sqrt((emb_q_ * emb_q_).sum(-1))))
        
        # eliminate negative terms
        weight = torch.where(weight>=0, weight,0)
        pseudo_label = labels.squeeze(-1)*weight

        return pseudo_label.reshape(-1)

    

    
        
        

