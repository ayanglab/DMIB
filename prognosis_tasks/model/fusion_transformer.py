import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones.densenet import DenseNet
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o

    
class TransformerFusion(nn.Module):
    def __init__(self, len_clinical=18, embed_dim=32, num_heads=4):
        super(TransformerFusion, self).__init__()
        
        self.img_backbone = DenseNet()
        self.clinical_backbone = nn.Sequential(nn.Linear(1, 256),
                                               nn.ReLU(),
                                               nn.Linear(256, 1024))
        self.multihead_attn = MultiheadAttention(len_clinical + 441, embed_dim, num_heads)
        self.classifier = nn.Sequential(nn.Linear(1024, 256),
                                        nn.Linear(256, 2))
        
        
    def forward(self, img, clinical):
        img_features = self.img_backbone.features(img)
        img_features = F.relu(img_features, inplace=True)
        img_features = img_features.view(img_features.shape[0], 1024, -1)
        clinical_features = self.clinical_backbone(clinical.unsqueeze(-1)).permute(0,2,1)
        concat = torch.cat((img_features, clinical_features), dim=-1)
        
        out = self.multihead_attn(concat)
        out = torch.mean(out, dim=-1)
        out = self.classifier(out)
        
        return out


class TransformerImage(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super(TransformerImage, self).__init__()
        
        self.img_backbone = DenseNet()
        self.multihead_attn = MultiheadAttention(441, embed_dim, num_heads)
        self.classifier = nn.Sequential(nn.Linear(1024, 256),
                                        nn.Linear(256, 2))
        
    def forward(self, img):
        img_features = self.img_backbone.features(img)
        img_features = F.relu(img_features, inplace=True)
        img_features = img_features.view(img_features.shape[0], 1024, -1)
        out = self.multihead_attn(img_features)
        out = torch.mean(out, dim=-1)
        out = self.classifier(out)
        
        return out