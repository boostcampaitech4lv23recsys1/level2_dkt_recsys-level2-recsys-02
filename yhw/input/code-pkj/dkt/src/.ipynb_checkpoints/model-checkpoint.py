import pdb

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)

class Feed_Forward_block(nn.Module):
    def __init__(self, dim_ff, drop):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim_ff)
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.norm1 = nn.LayerNorm(dim_ff)
        self.drop1 = nn.Dropout(drop)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.norm2 = nn.LayerNorm(dim_ff)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop1(self.layer1(self.norm0(x)))
        x = self.gelu(x)
        
        return self.gelu(self.drop2(self.norm2(self.layer2(x))))
    
def future_mask(seq_length, hidden_size):
    future_mask = np.triu(np.ones((64, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()
    def forward(self, input):

        test, question, tag, _, mask, interaction = input
        
        batch_size = interaction.size(0)
        
        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
            
        return out

class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.all_dims = sum(self.args.n_embdings.values())+3
        self.sql = args.max_seq_len
        
        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        #self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 2)
        self.embedding_cate = nn.Embedding(self.all_dims, self.hidden_dim // 2, padding_idx=0)
        
        # self.embedding_cate = nn.ModuleDict(
        #     {
        #         col: nn.Embedding(num + 1, self.hidden_dim // 2, padding_idx=0)
        #         for col, num in args.n_embdings.items()
        #     }
        # )
        
        num_cate_cols = len(args.cate_loc) + 1
        self.cate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim // 2 * num_cate_cols, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        print(self.hidden_dim // 2 * num_cate_cols)
        self.embedding_conti = nn.Sequential(
            nn.Linear(len(args.conti_loc), self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        
        
        self.lstm = nn.LSTM(
            self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
        )
        #self.layer_norm = nn.LayerNorm(self.args.hidden_size)
        
        self.mat = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.n_heads, 
                                               dropout=self.drop_out)
       # self.ffn_en = Feed_Forward_block(self.args.hidden_size) 
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.bert_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.args.hidden_size,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_size, 1)
        def get_reg(h, d):
            return nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.Dropout(d),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.Dropout(d),
            nn.ReLU(),
            nn.Linear(h, 1),            
        )        
        self.reg_layer = get_reg(self.args.hidden_size, self.drop_out)
        
        self.activation = nn.Sigmoid()
        
    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.args.hidden_size)
        h = h.to(self.args.device)

        c = torch.zeros(self.n_layers, batch_size, self.args.hidden_size)
        c = c.to(self.args.device)

        return (h, c)
    
    def forward(self, input):

        cate, conti, mask, interaction, _ = input
        batch_size = interaction.size(0)
        field_dims = np.array([0, *np.cumsum(list(self.args.n_embdings.values()))[:-1]]) + 2

        for i,k in enumerate(cate.keys()):
            cate[k] = cate[k] + field_dims[i]
            cate[k] = torch.where(cate[k] == field_dims[i], 0 , cate[k])
        c = interaction.unsqueeze(2)
        
        for index, i in enumerate(cate.values()):
            c = torch.cat([c, i.unsqueeze(2)], axis=2)
        
        # Embedding
#         embed_interaction = self.embedding_interaction(interaction)
        
#         embed_cate = [
#             embedding(cate[col_name])
#             for col_name, embedding in self.embedding_cate.items()
#         ]
#         embed_cate.insert(0, embed_interaction)
        
#         embed_cate = torch.cat(embed_cate, 2)

        embed_cate = self.embedding_cate(c).view((batch_size, self.sql, -1))

        embed_cate = self.cate_proj(embed_cate)
        
        cont_feats = torch.stack([col for col in conti.values()], 2)
        #print(cont_feats.shape, '11')
        embed_cont = self.embedding_conti(cont_feats)

        embed = torch.cat([embed_cate, embed_cont], 2)
        
        out, _ = self.lstm(embed)

        if self.args.pos:
            extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.args.bert_layers

            encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
            #print(len(encoded_layers), encoded_layers[-1].shape)
            sequence_output = encoded_layers[-1]
        else:
            att_mask = future_mask(out.size(1)).to('cuda')
            sequence_output, _ = self.mat((out)[-1:,:,:], out+embed, out+embed) #attn_mask=att_mask

    
#        sequence_output = sequence_output.contiguous().view(batch_size, -1, self.args.hidden_size)
        #out = self.layer_norm(sequence_output)
        #out = self.ffn_en(out)
        #out = self.fc(out).view(batch_size, -1)
        out = self.reg_layer(sequence_output[:, -1]).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
        if len(out) == 1:
            return out[0]
        return out.squeeze()

class LSTMATTN2(nn.Module):
    def __init__(self, args):
        super(LSTMATTN2, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.sql = args.max_seq_len
        
        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        
        self.embedding_cate = nn.ModuleDict(
            {
                col: nn.Embedding(num + 1, self.hidden_dim // 3, padding_idx=0)
                for col, num in args.n_embdings.items()
            }
        )
        
        self.args.hidden_size = self.n_heads * 31
        num_cate_cols = len(args.cate_loc) #+ 1
        self.cate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim // 3 * num_cate_cols, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )

        self.embedding_conti = nn.Sequential(
            nn.Linear(len(args.conti_loc), self.args.hidden_size // 2 + self.args.hidden_size % 2),
            nn.LayerNorm(self.args.hidden_size // 2 + self.args.hidden_size % 2),
        )
        
        
        self.lstm = nn.LSTM(
            self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
        )
        
        self.mat = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.n_heads, 
                                               dropout=self.drop_out)
        self.ffn_en = Feed_Forward_block(self.args.hidden_size, self.drop_out) 
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.bert_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.args.hidden_size,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_size, 1)
        
        self.activation = nn.Sigmoid()
    
    def forward(self, input):

        cate, conti, mask, interaction, _ = input
        batch_size = interaction.size(0)
        
        # Embedding 
        #embed_interaction = self.embedding_interaction(interaction)
        
        embed_cate = [
            embedding(cate[col_name])
            for col_name, embedding in self.embedding_cate.items()
        ]
        
        #embed_cate.insert(0, embed_interaction)
        
        embed_cate = torch.cat(embed_cate, 2)
        
        embed_cate = self.cate_proj(embed_cate)
        
        cont_feats = torch.stack([col for col in conti.values()], 2)
        embed_cont = self.embedding_conti(cont_feats)

        embed = torch.cat([embed_cate, embed_cont], 2)
        
        #out, _ = self.lstm(embed)
        #print(embed.shape)
        if self.args.pos:
            extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.args.bert_layers

            encoded_layers = self.attn(embed, extended_attention_mask, head_mask=head_mask)
            sequence_output = encoded_layers[-1] 
        else:
            #att_mask = future_mask(embed.size(1), embed.size(2)).to('cuda')
            sequence_output, _ = self.mat(embed, embed, embed)
        #print(sequence_output.shape)
        out = self.ffn_en(sequence_output+embed)
        #print(out.shape)
        out = self.fc(out).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)

        return out#.view(-1)
    
class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
            
        return out

class Bert2(nn.Module):
    def __init__(self, args):
        super(Bert2, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        #self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.cate = nn.Embedding(14310, self.hidden_dim // 3)
        self.args.hidden_size = self.args.hidden_size - self.args.hidden_size%self.args.n_heads 
        # embedding combination projection
        num_fe_cols = len(args.cate_feats)
        self.embedding_cate = nn.Sequential(
            nn.Linear(self.hidden_dim // 3 * num_fe_cols, self.args.hidden_size // 2 + self.args.hidden_size % 2),
            nn.LayerNorm(self.args.hidden_size // 2 + self.args.hidden_size % 2),
        )
        
        self.embedding_conti = nn.Sequential(
            nn.Linear(len(args.conti_feats), self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        # Bert config
        self.attn = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.args.n_heads, 
                                               dropout=self.args.drop_out)

        self.attn2 = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.args.n_heads, 
                                               dropout=self.args.drop_out)
        # Fully connected layer
        self.ffe = Feed_Forward_block(self.args.hidden_size, self.args.drop_out)
        self.ffe2 = Feed_Forward_block(self.args.hidden_size, self.args.drop_out)
        self.fc = nn.Linear(self.args.hidden_size, 1)
        self.lstm = nn.LSTM(
            self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
        )
        self.activation = nn.Sigmoid()

    def forward(self, cate, conti, answer):
        #cate, conti, answer = input
        batch_size = cate.size(0)
        seq_len = cate.size(1)
        # 신나는 embedding
        cate = self.cate(cate).view(batch_size, seq_len, -1)
        #print(cate.shape, self.args.allF, self.embedding_cate)
        cate = self.embedding_cate(cate)
        
        conti = self.embedding_conti(conti)
        
        embed = torch.cat([cate, conti], 2)
        # Bert
        
        out, _ = self.attn(embed, embed, embed)
        #head_mask = [None] * self.args.bert_layers
        
        #encoded_layers = self.attn(embed, extended_attention_mask, head_mask=head_mask)
        #out = self.encoder(embed, head_mask=head_mask)[0]
        #out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out2 = self.ffe(out+embed)
        if self.args.bert_layers == 2:
            out3, _ = self.attn(out2, out2, out2)
            out = self.ffe(out3 + out2)
            out = self.fc(out3 + out).view(batch_size, -1)
        else:
            #out, _ = self.lstm(out2+out)
            out = self.fc(out+out2).view(batch_size, -1)
            
        if self.args.isinfer:
            out = self.activation(out)
            
        return out
    
class ATTNLSTM(nn.Module):
    def __init__(self, args):
        super(ATTNLSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.all_dims = sum(self.args.n_embdings.values())+3
        self.sql = args.max_seq_len
        
        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        #self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 2)
        self.embedding_cate = nn.Embedding(self.all_dims, self.hidden_dim // 2, padding_idx=0)
        
        # self.embedding_cate = nn.ModuleDict(
        #     {
        #         col: nn.Embedding(num + 1, self.hidden_dim // 2, padding_idx=0)
        #         for col, num in args.n_embdings.items()
        #     }
        # )
        
        num_cate_cols = len(args.org_feats) #len(args.cate_loc) + 1
        self.cate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim // 2 * num_cate_cols, self.args.hidden_size // 8 * 7),
            nn.LayerNorm(self.args.hidden_size // 8 * 7),
        )
        
        self.embedding_conti = nn.Sequential(
            nn.Linear(1, self.args.hidden_size // 8 + self.args.hidden_size % 8),
            nn.LayerNorm(self.args.hidden_size // 8 + self.args.hidden_size % 8),
        )
        num_fe_cols = len(args.fe_feats)+1
        
        self.fe_cate = nn.Sequential(
            nn.Linear(self.hidden_dim // 2 * num_fe_cols, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        
        self.fe_conti = nn.Sequential(
            nn.Linear(len(args.conti_feats)-1, self.args.hidden_size // 2 + self.args.hidden_size % 2),
            nn.LayerNorm(self.args.hidden_size // 2 + self.args.hidden_size % 2),
        )
        
        self.lstm = nn.LSTM(
            self.args.hidden_size*2, self.args.hidden_size*2, self.n_layers, batch_first=True
        )
        #self.layer_norm = nn.LayerNorm(self.args.hidden_size)
        
        self.mat = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.n_heads, 
                                               dropout=self.drop_out)
       # self.ffn_en = Feed_Forward_block(self.args.hidden_size) 
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.bert_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.args.hidden_size,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_size, 1)
        def get_reg(h, d):
            return nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.Dropout(d),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.Dropout(d),
            nn.ReLU(),
            nn.Linear(h, 1),            
        )        
        self.reg_layer = get_reg(self.args.hidden_size*2, self.drop_out)
        
        self.activation = nn.Sigmoid()
        
    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.args.hidden_size)
        h = h.to(self.args.device)

        c = torch.zeros(self.n_layers, batch_size, self.args.hidden_size)
        c = c.to(self.args.device)

        return (h, c)
    
    def forward(self, input):

        cate, conti, mask, interaction, _ = input
        batch_size = interaction.size(0)
        field_dims = np.array([0, *np.cumsum(list(self.args.n_embdings.values()))[:-1]]) + 2
        
        for i,k in enumerate(cate.keys()):
            cate[k] = cate[k] + field_dims[i]
            cate[k] = torch.where(cate[k] == field_dims[i], 0 , cate[k])
        interaction = interaction.unsqueeze(2)
        #c = interaction.unsqueeze(2)
        #c = torch.tensor([])
        c = torch.stack([cate[col] for col in self.args.org_feats], 2)
        #c = torch.cat([interaction, c], axis=2)
        fe = torch.stack([cate[col] for col in self.args.fe_feats], 2)
        #print(fe.shape, len(self.args.fe_feats))
        fe = torch.cat([interaction, fe], axis=2)
        #print(fe.shape, interaction.shape)
        # for index, i in enumerate(cate.values()):
        #     c = torch.cat([c, i.unsqueeze(2)], axis=2)
        
        
        # Embedding
#         embed_interaction = self.embedding_interaction(interaction)
        
#         embed_cate = [
#             embedding(cate[col_name])
#             for col_name, embedding in self.embedding_cate.items()
#         ]
#         embed_cate.insert(0, embed_interaction)
        
#         embed_cate = torch.cat(embed_cate, 2)
        embed_cate = self.embedding_cate(c).view((batch_size, self.sql, -1))
        embed_fe = self.embedding_cate(fe).view((batch_size, self.sql, -1))
        #print(embed_fe.shape)
        embed_cate = self.cate_proj(embed_cate)
        
        # for index, i in enumerate(cate.values()):
        #     c = torch.cat([c, i.unsqueeze(2)], axis=2)
            
        cont_feats = conti['Time'].unsqueeze(2) # torch.stack([col for col in conti.values()], 2)
        
        embed_cont = self.embedding_conti(cont_feats)

        embed = torch.cat([embed_cate, embed_cont], 2)
        
        

        if self.args.pos:
            extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.args.bert_layers

            encoded_layers = self.attn(embed, extended_attention_mask, head_mask=head_mask)
            sequence_output = encoded_layers[-1]
        else:
            sequence_output, _ = self.mat(embed[-1:,:,:], embed, embed)
        #print(embed_fe.shape)
        embed_fe = self.fe_cate(embed_fe)
        cont_fe = torch.stack([conti[col] for col in conti.keys() if col != 'Time'], 2)
        cont_fe = self.fe_conti(cont_fe)
        
        fe_embed = torch.cat([embed_fe, cont_fe], 2)
        ipt = torch.cat([sequence_output, fe_embed], 2)
        out, _ = self.lstm(ipt)
#        sequence_output = sequence_output.contiguous().view(batch_size, -1, self.args.hidden_size)
        #out = self.layer_norm(sequence_output)
        #out = self.ffn_en(out)
        #out = self.fc(out).view(batch_size, -1)
        #print(sum(out[:, -1], 1).shape)
        #return torch.sum(out[:, -1], 1)
        out = self.reg_layer(out[:, -1]).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
        
        if len(out) == 1:
            return out[0]
        return out.squeeze()
    
    

class ATTNLSTM2(nn.Module):
    def __init__(self, args):
        super(ATTNLSTM2, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.all_dims = sum(self.args.n_embdings.values())+3
        self.sql = args.max_seq_len
        
        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        #self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 2)
        self.embedding_cate = nn.Embedding(self.all_dims, self.hidden_dim // 2, padding_idx=0)
        
        # self.embedding_cate = nn.ModuleDict(
        #     {
        #         col: nn.Embedding(num + 1, self.hidden_dim // 2, padding_idx=0)
        #         for col, num in args.n_embdings.items()
        #     }
        # )
        
        num_cate_cols = len(args.org_feats) #len(args.cate_loc) + 1
        self.cate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim // 2 * num_cate_cols, self.args.hidden_size // 8 * 7),
            nn.LayerNorm(self.args.hidden_size // 8 * 7),
        )
        
        self.embedding_conti = nn.Sequential(
            nn.Linear(1, self.args.hidden_size // 8),
            nn.LayerNorm(self.args.hidden_size // 8),
        )
        num_fe_cols = len(args.fe_feats)+1
        
        self.fe_cate = nn.Sequential(
            nn.Linear(self.hidden_dim // 2 * num_fe_cols, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        
        self.fe_conti = nn.Sequential(
            nn.Linear(len(args.conti_feats)-1, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        
        self.lstm = nn.LSTM(
            self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
        )
        #self.layer_norm = nn.LayerNorm(self.args.hidden_size)
        
        self.mat = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.n_heads, 
                                               dropout=self.drop_out)
        # self.ff = Feed_Forward_block(self.args.hidden_size)
        # self.mat2 = torch.nn.MultiheadAttention(self.args.hidden_size,
        #                                num_heads=self.n_heads, 
        #                                dropout=self.drop_out)
       # self.ffn_en = Feed_Forward_block(self.args.hidden_size) 
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.bert_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.args.hidden_size,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_size, 1)
        def get_reg(h, d):
            return nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.Dropout(d),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.Dropout(d),
            nn.ReLU(),
            nn.Linear(h, 1),            
        )        
        self.reg_layer = get_reg(self.args.hidden_size, self.drop_out)
        
        self.activation = nn.Sigmoid()
        
    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.args.hidden_size)
        h = h.to(self.args.device)

        c = torch.zeros(self.n_layers, batch_size, self.args.hidden_size)
        c = c.to(self.args.device)

        return (h, c)
    
    def forward(self, input):

        cate, conti, mask, interaction, _ = input
        batch_size = interaction.size(0)
        field_dims = np.array([0, *np.cumsum(list(self.args.n_embdings.values()))[:-1]]) + 2
        
        for i,k in enumerate(cate.keys()):
            cate[k] = cate[k] + field_dims[i]
            cate[k] = torch.where(cate[k] == field_dims[i], 0 , cate[k])
        interaction = interaction.unsqueeze(2)
        #c = interaction.unsqueeze(2)
        #c = torch.tensor([])
        c = torch.stack([cate[col] for col in self.args.org_feats], 2)
        #c = torch.cat([interaction, c], axis=2)
        fe = torch.stack([cate[col] for col in self.args.fe_feats], 2)
        #print(fe.shape, len(self.args.fe_feats))
        fe = torch.cat([interaction, fe], axis=2)
        #print(fe.shape, interaction.shape)
        # for index, i in enumerate(cate.values()):
        #     c = torch.cat([c, i.unsqueeze(2)], axis=2)
        
        
        # Embedding
#         embed_interaction = self.embedding_interaction(interaction)
        
#         embed_cate = [
#             embedding(cate[col_name])
#             for col_name, embedding in self.embedding_cate.items()
#         ]
#         embed_cate.insert(0, embed_interaction)
        
#         embed_cate = torch.cat(embed_cate, 2)
        embed_cate = self.embedding_cate(c).view((batch_size, self.sql, -1))
        embed_fe = self.embedding_cate(fe).view((batch_size, self.sql, -1))
        #print(embed_fe.shape)
        embed_cate = self.cate_proj(embed_cate)
        
        # for index, i in enumerate(cate.values()):
        #     c = torch.cat([c, i.unsqueeze(2)], axis=2)
            
        cont_feats = conti['Time'].unsqueeze(2) # torch.stack([col for col in conti.values()], 2)
        
        embed_cont = self.embedding_conti(cont_feats)

        embed = torch.cat([embed_cate, embed_cont], 2)
        
        

        if self.args.pos:
            extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.args.bert_layers

            encoded_layers = self.attn(embed, extended_attention_mask, head_mask=head_mask)
            sequence_output = encoded_layers[-1]
        else:
            sequence_output, _ = self.mat(embed[-1:,:,:], embed, embed)
            # sequence_output = self.ff(sequence_output)
            # sequence_output, _ = self.mat2(sequence_output[-1:,:,:], sequence_output, sequence_output)
            
        #print(embed_fe.shape)
        embed_fe = self.fe_cate(embed_fe)
        cont_fe = torch.stack([conti[col] for col in conti.keys() if col != 'Time'], 2)
        cont_fe = self.fe_conti(cont_fe)
        
        fe_embed = torch.cat([embed_fe, cont_fe], 2)
        # ipt = torch.cat([sequence_output, fe_embed], 2)
        out, _ = self.lstm(sequence_output+fe_embed)
#        sequence_output = sequence_output.contiguous().view(batch_size, -1, self.args.hidden_size)
        #out = self.layer_norm(sequence_output)
        #out = self.ffn_en(out)
        #out = self.fc(out).view(batch_size, -1)
        #print(sum(out[:, -1], 1).shape)
        #return torch.sum(out[:, -1], 1)
        out = self.reg_layer(out[:, -1]).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
        
        if len(out) == 1:
            return out[0]
        return out.squeeze()
    
# class ATTNLSTM(nn.Module):
#     def __init__(self, args):
#         super(ATTNLSTM, self).__init__()
#         self.args = args

#         self.hidden_dim = self.args.hidden_dim
#         self.n_layers = self.args.n_layers
#         self.n_heads = self.args.n_heads
#         self.drop_out = self.args.drop_out
        
#         # Embedding
#         # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
#         self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 2)
#         #self.embedding_interaction2 = nn.Embedding(3, self.hidden_dim)
        
#         self.embedding_cate = nn.ModuleDict(
#             {
#                 col: nn.Embedding(num + 1, self.hidden_dim // 2)
#                 for col, num in args.n_embdings.items()
#             }
#         )
        
#         # 1d-cnn -> 
#         # 1d-cnn ->
#         # 1d-cnn ->
        
#         num_cate_cols = len(args.cate_loc) + 1
#         self.cate_proj = nn.Sequential(
#             nn.Linear(self.hidden_dim // 2 * num_cate_cols, self.args.hidden_size // 2),
#             nn.LayerNorm(self.args.hidden_size // 2),
#         )
        
#         self.embedding_conti = nn.Sequential(
#             nn.Linear(len(args.conti_loc), self.args.hidden_size // 2),
#             nn.LayerNorm(self.args.hidden_size // 2),
#         )
        
#         self.lstm = nn.LSTM(
#             self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
#         )
#         self.layer_norm1 = nn.LayerNorm(self.hidden_dim // 2)
#         self.layer_norm2 = nn.LayerNorm(self.hidden_dim // 2)
        
#         self.mat = torch.nn.MultiheadAttention(self.hidden_dim // 2,
#                                                num_heads=self.args.bert_layers, 
#                                                dropout=self.drop_out)
#         self.ffn_en = Feed_Forward_block(self.hidden_dim // 2) 
#         self.config = BertConfig(
#             3,  # not used
#             hidden_size=self.args.hidden_size,
#             num_hidden_layers=self.args.bert_layers,
#             num_attention_heads=self.n_heads,
#             intermediate_size=self.args.hidden_size,
#             hidden_dropout_prob=self.drop_out,
#             attention_probs_dropout_prob=self.drop_out,
#         )
#         self.attn = BertEncoder(self.config)

#         # Fully connected layer
#         self.fc = nn.Linear(self.args.hidden_size, 1)

#         self.activation = nn.Sigmoid()

#     def forward(self, input):

#         cate, conti, mask, interaction, _, = input

#         batch_size = interaction.size(0)

#         # Embedding
#         embed_interaction = self.embedding_interaction(interaction) # 이전 문제를 맞았나 틀렸나
#         #embed_interaction2 = self.embedding_interaction2(interaction2) # 이전의 이전
        
#         embed_cate = [
#             embedding(cate[col_name])
#             for col_name, embedding in self.embedding_cate.items()
#         ]
#         # nn.embedding([1,2,3]) // nn.embedding([1]) nn.embedding([2]) nn.embedding([3])
#         # nn.embedding([1,2,3]) // 
#         # lstm attn
#         # lstm -> f1 f2 f3 //f1 f2 f3 //f1 f2 f3 // dfsif
#         # attn,lstm // lstm,attn
#         # 
#         embed_cate.insert(0, embed_interaction)
#         #embed_cate.insert(1, embed_interaction2)
#         #print(len(embed_cate), embed_cate[0].shape, '1')
#         embed_cate = torch.cat(embed_cate, 2)
        
#         #print(embed_cate.shape, '2')
#         embed_cate = self.cate_proj(embed_cate)
        
#         cont_feats = torch.stack([col for col in conti.values()], 2)
#         embed_cont = self.embedding_conti(cont_feats)

#         embed = torch.cat([embed_cate, embed_cont], 2)
#         #print(embed_cate.shape, embed_cont.shape, embed.shape, self.args.hidden_size)
#         if self.args.pos:
#             extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
#             extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
#             extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#             head_mask = [None] * self.args.bert_layers

#             encoded_layers = self.attn(embed, extended_attention_mask, head_mask=head_mask)
#             sequence_output = encoded_layers[-1]
#         else:
#             attension_output, _ = self.mat(embed[-1:,:,:], embed, embed)
#             sequence_output = self.layer_norm1(sequence_output)
            
#             sequence_output = self.layer_norm2(self.ffn_en(sequence_output) + sequence_output)
            
#         out, _ = self.lstm(sequence_output)
        
#         out = out.contiguous().view(batch_size, -1, self.args.hidden_size)
#         # out = self.layer_norm(out) + sequence_output
#         # out = self.ffn_en(out)
#         out = self.fc(out).view(batch_size, -1)
        
#         if self.args.isinfer:
#             out = self.activation(out)
            
#         return out
    
    
# class ATTNLSTM2(nn.Module):
#     def __init__(self, args):
#         super(ATTNLSTM2, self).__init__()
#         self.args = args

#         self.hidden_dim = self.args.hidden_dim
#         self.n_layers = self.args.n_layers
#         self.n_heads = self.args.n_heads
#         self.drop_out = self.args.drop_out
        
#         # Embedding
#         # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
#         self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 2)
#         #self.embedding_interaction2 = nn.Embedding(3, self.hidden_dim)
        
#         self.embedding_cate = nn.ModuleDict(
#             {
#                 col: nn.Embedding(num + 1, self.hidden_dim // 2)
#                 for col, num in args.n_embdings.items()
#             }
#         )
        
#         num_cate_cols = len(args.cate_loc) + 1
#         self.cate_proj = nn.Sequential(
#             nn.Linear(self.hidden_dim // 2 * num_cate_cols, self.args.hidden_size),
#             nn.LayerNorm(self.args.hidden_size),
#             #nn.ReLU()
#         )
#         self.lstm_input = nn.Linear(self.hidden_dim // 2 * num_cate_cols, self.args.hidden_size)
        
#         self.embedding_conti = nn.Sequential(
#             nn.Linear(len(args.conti_loc), self.args.hidden_size // 2),
#             nn.LayerNorm(self.args.hidden_size // 2),
#             #nn.ReLU()
#         )
        
#         self.lstm = nn.LSTM(
#             self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
#         )
#         self.layer_norm1 = nn.LayerNorm(self.hidden_dim // 2)
#         self.layer_norm2 = nn.LayerNorm(self.hidden_dim // 2)
        
#         self.mat = torch.nn.MultiheadAttention(self.hidden_dim // 2,
#                                                num_heads=self.args.bert_layers, 
#                                                dropout=self.drop_out)
#         self.ffn_en = Feed_Forward_block(self.hidden_dim // 2) 
#         self.config = BertConfig(
#             3,  # not used
#             hidden_size=self.args.hidden_size,
#             num_hidden_layers=self.args.bert_layers,
#             num_attention_heads=self.n_heads,
#             intermediate_size=self.args.hidden_size,
#             hidden_dropout_prob=self.drop_out,
#             attention_probs_dropout_prob=self.drop_out,
#         )
#         self.attn = BertEncoder(self.config)

#         # Fully connected layer
#         self.fc = nn.Linear(self.args.hidden_size, 1)

#         self.activation = nn.Sigmoid()

#     def forward(self, input):

#         cate, conti, mask, interaction, _, = input

#         batch_size = interaction.size(0)

#         # Embedding
#         embed_interaction = self.embedding_interaction(interaction) # 이전 문제를 맞았나 틀렸나
#         #embed_interaction2 = self.embedding_interaction2(interaction2) # 이전의 이전
        
#         embed_cate = [
#             embedding(cate[col_name])
#             for col_name, embedding in self.embedding_cate.items()
#         ]
#         embed_cate.insert(0, embed_interaction)
#         #embed_cate.insert(1, embed_interaction2)
#         #print(len(embed_cate), embed_cate[0].shape, '1')
#         embed_cate = torch.cat(embed_cate, 2)
#         embed = embed_cate.view([-1, self.args.max_seq_len, len(self.args.cate_loc) + 1, self.hidden_dim // 2])
        
#         #print(embed_cate.shape, '2')
#         embed = self.cate_proj(embed_cate)
        
#         cont_feats = torch.stack([col for col in conti.values()], 2)
#         embed_cont = self.embedding_conti(cont_feats)

#         embed = torch.cat([embed_cate, embed_cont], 2)
       
#         if self.args.pos:
#             extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
#             extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
#             extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#             head_mask = [None] * self.args.bert_layers

#             encoded_layers = self.attn(embed, extended_attention_mask, head_mask=head_mask)
#             sequence_output = encoded_layers[-1]
#         else:
#             sequence_output = []
#             for i in range(len(embed)):
#                 attension_output, _ = self.mat(embed[-1,:,:], embed[i,:,:], embed[i,:,:])
#                 sequence_output.append(attension_output)
#             sequence_output = torch.stack([col for col in attension_output], 0) + embed
                                
#         sequence_output = self.layer_norm1(sequence_output)
#         out = self.layer_norm2(self.ffn_en(sequence_output) + sequence_output)
#         out = out.view([-1, self.args.max_seq_len, self.hidden_dim // 2 * (len(self.args.cate_loc) + 1)])
#         out = self.lstm_input(out)
#         out, _ = self.lstm(out)
        
#         out = out.contiguous().view(batch_size, -1, self.args.hidden_size)
#         # out = self.layer_norm(out) + sequence_output
#         # out = self.ffn_en(out)
#         out = self.fc(out).view(batch_size, -1)
        
#         if self.args.isinfer:
#             out = self.activation(out)
            
#         return out