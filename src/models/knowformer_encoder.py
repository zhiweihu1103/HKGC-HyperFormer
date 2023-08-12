import torch
import torch.nn as nn
import math
import copy
from scipy.stats import truncnorm
from fmoe import FMoETransformerMLP
from torch.nn import Parameter

def truncated_normal(size, threshold=0.02):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def truncated_normal_init(x, size, initializer_range):
    x.weight.data.copy_(
        torch.from_numpy(truncated_normal(size, initializer_range)))
    if hasattr(x, "bias"):
        nn.init.constant_(x.bias, 0.0)

def truncated_normal_init_param(x, size, initializer_range):
    x.data.copy_(
        torch.from_numpy(truncated_normal(size, initializer_range)))
    if hasattr(x, "bias"):
        nn.init.constant_(x.bias, 0.0)

def norm_layer_init(x):
    nn.init.constant_(x.weight, 1.0)  # init
    nn.init.constant_(x.bias, 0.0)  # init

def clones(module, n):
    module_list = nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
    if isinstance(module, nn.Linear):
        for i in range(0, n):
            module_list[i].__init__(module.in_features, module.out_features)
    else:
        for i in range(0, n):
            module_list[i].__init__(module.size, module.residual_dropout_prob)
    return module_list


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, initializer_range):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        truncated_normal_init(self.lut, [vocab, d_model], initializer_range)

    def forward(self, x):
        out = self.lut(x)
        return out

def get_param(shape, norm=False):
    param = Parameter(torch.Tensor(*shape))
    truncated_normal_init_param(param, [shape[0], shape[1]], 0.02)
    # nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
    return param

def attention(query, key, value, mask, attention_dropout_layer):
    d_k = query.size(-1)
    
    # scores.shape: (batch_size, head_size, seq_size=3, seq_size=3)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn.shape: (batch_size, head_size, seq_size, seq_size)
    p_attn = nn.functional.softmax(scores, dim=-1)
    
    p_attn = attention_dropout_layer(p_attn)
    out = torch.matmul(p_attn, value)
    return out, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, attention_dropout_prob=0.1, initializer_range=0.02):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        for linear in self.linears:
            truncated_normal_init(linear, [d_model, d_model], initializer_range)
        self.attn = None
        self.attention_dropout_layer = nn.Dropout(p=attention_dropout_prob)

    def forward(self, query, key, value, mask):
        # query, key, value shape: (batch_size,seq_size,d_model)
        # mask.shape: (batch_size,head_size,seq_size,seq_size)

        nbatches = query.size(0)

        # l(x) shape: (batch_size,seq_size,head_size,feature_size=d_model//head_size)
        # query, key, value shape: (batch_size,head_size,seq_size,feature_size=d_model//head_size)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).permute(0, 2, 1, 3)
             for l, x in zip(self.linears, (query, key, value))]

        # x.shape: (batch_size,head_size,seq_size,feature_size)
        # self.attn: (batch_size,head_size,seq_size,seq_size)
        x, self.attn = attention(query, key, value, mask=mask,
                                 attention_dropout_layer=self.attention_dropout_layer)

        # x.shape: (batch_size,head_size,seq_size,feature_size) -> (batch_size,seq_size,d_model)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # out.shape: (batch_size,seq_size,d_model)
        out = self.linears[-1](x)
        return out


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, moe_num_expert=64, moe_top_k=2):
        activation = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        super().__init__(num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, top_k=moe_top_k,
                activation=activation)

        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, hidden_dropout_prob=0.1, initializer_range=0.02):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        truncated_normal_init(self.w_1, [d_ff, d_model], initializer_range)

        self.w_2 = nn.Linear(d_ff, d_model)
        truncated_normal_init(self.w_2, [d_model, d_ff], initializer_range)

        self.hidden_dropout_layer = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        # x.shape: (batch_size,seq_size=3,d_model)
        # return.shape: (batch_size,seq_size=3,d_model)
        return self.w_2(self.hidden_dropout_layer(nn.functional.gelu(self.w_1(x))))
        

class SublayerConnection(nn.Module):
    def __init__(self, size, residual_dropout_prob):
        super(SublayerConnection, self).__init__()
        self.size = size
        self.residual_dropout_prob = residual_dropout_prob
        self.norm = nn.LayerNorm(size)
        norm_layer_init(self.norm)
        self.residual_dropout_layer = nn.Dropout(self.residual_dropout_prob)

    def forward(self, x, sublayer):
        return x + self.residual_dropout_layer(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config, self_attn, feed_forward, size, residual_dropout_prob, moe_mode):
        super(EncoderLayer, self).__init__()
        self.config = config
        self.moe_mode = moe_mode
        self.self_attn = self_attn
        self.size = size
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, residual_dropout_prob), 2)
        self.norm1 = nn.LayerNorm(config['hidden_size'])
        self.norm2 = nn.LayerNorm(config['hidden_size'])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        if self.moe_mode == 'False':
            out = self.sublayer[1](x, self.feed_forward)
        else:
            out = self.feed_forward(x)
        return out

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Encoder(nn.Module):
    def __init__(self, config, moe_mode=False):
        super(Encoder, self).__init__()
        self.config = config
        self.moe_mode = moe_mode
        self.pooler = BertPooler(config['hidden_size'])

        layers = []
        for i in range(config['num_hidden_layers']):
            attn = MultiHeadedAttention(config['num_attention_heads'], config['hidden_size'],
                                        config['attention_dropout_prob'], config['initializer_range'])
            if moe_mode == 'False':
                ff = PositionwiseFeedForward(config['hidden_size'], config['intermediate_size'],
                                             config['hidden_dropout_prob'], config['initializer_range'])
            else:
                ff = CustomizedMoEPositionwiseFF(config['hidden_size'], config['intermediate_size'] // config['moe_top_k'],
                                             config['hidden_dropout_prob'], True, config['moe_num_expert'], config['moe_top_k'])
            layer = EncoderLayer(config, attn, ff, config['hidden_size'], config['residual_dropout_prob'], self.moe_mode)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(self.layers[0].size)
        norm_layer_init(self.norm)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        
    def forward(self, x, mask):
        # x.shape: (batch_size, seq_size=3, d_model)
        # mask.shape: (batch_size, seq_size=3, seq_size=3)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x, mask)
        return self.norm(x)