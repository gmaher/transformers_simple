import torch
from torch.nn import functional as F
import numpy as np

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, block_size,
    attn_dropout=0.2, residual_dropout=0.2, activation=torch.nn.Identity()):
        super(MultiHeadAttention,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.activation = activation

        self.Wa = torch.nn.Linear(input_size, num_heads*3*hidden_size, bias=False)

        self.Wout = torch.nn.Linear(hidden_size*num_heads, output_size)

        self.mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(
        1,1,self.block_size,self.block_size
        )

        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)
        self.residual_dropout = torch.nn.Dropout(p=residual_dropout)

    def forward(self, x):
        """
            x - (Nbatch, Length, Vec size)
        """
        nb, l, e = x.size()

        tmp = self.Wa(x)
        tmp = tmp.view(nb, l, self.num_heads, 3*self.hidden_size).transpose(1,2)
        #tmp is now (nb, num_heads, l, 3*hidden_size)

        q,k,v = torch.split(tmp, self.hidden_size, dim=3)

        # A is (nb, num_heads, l, l)
        A = q @ k.transpose(-2,-1)/np.sqrt(self.hidden_size)
        A = A.masked_fill(self.mask[:,:,:l,:l] ==0, float('-inf'))
        A = F.softmax(A,dim=-1)
        A = self.attn_dropout(A)

        o = A @ v #(nb, num_heads, l, hidden_size)

        o = o.transpose(1,2).contiguous().view(nb,l,self.hidden_size*self.num_heads)
        o = self.Wout(o)

        o = self.activation(o)
        o = self.residual_dropout(o)

        return o

class TransformerBlock(torch.nn.Module):
    def __init__(self, block_size, vec_size, hidden_size, attn_hidden_size,
    output_size, num_heads, attn_dropout=0.2, attn_residual_dropout=0.2,
    dropout=0.2, activation=torch.nn.Identity()):

        super(TransformerBlock, self).__init__()

        self.block_size = block_size
        self.vec_size = vec_size
        self.hidden_size = hidden_size
        self.attn_hidden_size = attn_hidden_size
        self.attn_output_size = vec_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.activation = activation

        self.attn_module = MultiHeadAttention(input_size=self.vec_size,
        hidden_size=self.attn_hidden_size,
        output_size=self.output_size,
        num_heads=self.num_heads,
        block_size=self.block_size,
        attn_dropout=attn_dropout,
        residual_dropout=attn_residual_dropout)

        self.fc1 = torch.nn.Linear(self.attn_output_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

        self.norm1 = torch.nn.LayerNorm(self.vec_size)
        self.norm2 = torch.nn.LayerNorm(self.output_size)

        self.dropout=torch.nn.Dropout(p=dropout)

    def forward(self,x):
        """
        args:
            x - (Nbatch, block_size, vec_size)


        """
        o = self.norm1(x)
        o = self.attn_module(o)
        o = self.norm2(x+o)
        o = self.fc1(o)
        o = self.activation(o)
        o = x+self.fc2(o)
        o = self.dropout(o)
        return o

class GPT(torch.nn.Module):
    def __init__(self, vocab_size, block_size, embed_size, hidden_size,
    attn_hidden_size, output_size, num_transformer_blocks, num_heads,
    embed_dropout=0.2, activation=torch.nn.LeakyReLU(0.05)):
        super(GPT, self).__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_size = embed_size
        self.attn_hidden_size = attn_hidden_size
        self.output_size = output_size
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.activation = activation

        self.embedder = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        self.embed_dropout = torch.nn.Dropout(embed_dropout)

        self.transformer_blocks = torch.nn.ModuleList()
        for n in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(block_size=block_size, vec_size=embed_size,
                hidden_size=hidden_size, attn_hidden_size=attn_hidden_size,
                output_size=embed_size,
                num_heads=num_heads, activation=activation)
            )

        self.norm_out = torch.nn.LayerNorm(embed_size)
        self.fc_out = torch.nn.Linear(embed_size, output_size, bias=False)

    def forward(self,x):
        """
            x - (Nbatch, block_size), vectors of vocab ids
        """

        o = self.embedder(x) #now (Nbatch, block_size, embed_size)
        o = self.embed_dropout(o)
        for tb in self.transformer_blocks:
            o = tb(o)

        o = self.norm_out(o)
        logits = self.fc_out(o) #now (Nbatch, block_size, vocab_size)

        return logits
