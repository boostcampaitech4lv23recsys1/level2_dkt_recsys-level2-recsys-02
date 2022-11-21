import torch
import torch.nn as nn
import pdb

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


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

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        
        self.embedding_cate = nn.ModuleDict(
            {
                col: nn.Embedding(num + 1, self.hidden_dim)
                for col, num in args.n_embdings.items()
            }
        )
        
        num_cate_cols = len(args.cate_loc) + 1
        self.cate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * num_cate_cols, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        
        self.embedding_conti = nn.Sequential(
            nn.Linear(len(args.conti_loc), self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
        )
        
        self.lstm = nn.LSTM(
            self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.args.hidden_size)
        
        self.mat = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.args.bert_layers, 
                                               dropout=self.drop_out)
        self.ffn_en = Feed_Forward_block(self.args.hidden_size) 
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
        embed_interaction = self.embedding_interaction(interaction)
        
        embed_cate = [
            embedding(cate[col_name])
            for col_name, embedding in self.embedding_cate.items()
        ]
        embed_cate.insert(0, embed_interaction)
        
        embed_cate = torch.cat(embed_cate, 2)

        embed_cate = self.cate_proj(embed_cate)
        
        cont_feats = torch.stack([col for col in conti.values()], 2)
        embed_cont = self.embedding_conti(cont_feats)

        embed = torch.cat([embed_cate, embed_cont], 2)
        
        out, _ = self.lstm(embed)

        if self.args.pos:
            extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = [None] * self.args.bert_layers

            encoded_layers = self.attn(out+embed, extended_attention_mask, head_mask=head_mask)
            sequence_output = encoded_layers[-1]
        else:
            sequence_output, _ = self.mat((out+embed)[-1:,:,:], out+embed, out+embed)

    
#        sequence_output = sequence_output.contiguous().view(batch_size, -1, self.args.hidden_size)
        out = self.layer_norm(sequence_output) + out
        out = self.ffn_en(out)
        out = self.fc(out).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
            
        return out


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

class ATTNLSTM(nn.Module):
    def __init__(self, args):
        super(ATTNLSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        #self.embedding_interaction2 = nn.Embedding(3, self.hidden_dim)
        
        self.embedding_cate = nn.ModuleDict(
            {
                col: nn.Embedding(num + 1, self.hidden_dim)
                for col, num in args.n_embdings.items()
            }
        )
        
        num_cate_cols = len(args.cate_loc) + 1 #+ 1
        self.cate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * num_cate_cols, self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
            #nn.ReLU()
        )
        
        self.embedding_conti = nn.Sequential(
            nn.Linear(len(args.conti_loc), self.args.hidden_size // 2),
            nn.LayerNorm(self.args.hidden_size // 2),
            #nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            self.args.hidden_size, self.args.hidden_size, self.n_layers, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.args.hidden_size)
        
        self.mat = torch.nn.MultiheadAttention(self.args.hidden_size,
                                               num_heads=self.args.bert_layers, 
                                               dropout=self.drop_out)
        self.ffn_en = Feed_Forward_block(self.args.hidden_size) 
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

        cate, conti, mask, interaction, _, = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction) # 이전 문제를 맞았나 틀렸나
        #embed_interaction2 = self.embedding_interaction2(interaction2)
        
        embed_cate = [
            embedding(cate[col_name])
            for col_name, embedding in self.embedding_cate.items()
        ]
        embed_cate.insert(0, embed_interaction)
        #embed_cate.insert(1, embed_interaction2)
        
        embed_cate = torch.cat(embed_cate, 2)

        embed_cate = self.cate_proj(embed_cate)
        
        cont_feats = torch.stack([col for col in conti.values()], 2)
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
            
        out, _ = self.lstm(sequence_output+embed)
        
        out = out.contiguous().view(batch_size, -1, self.args.hidden_size)
        out = self.layer_norm(out) + sequence_output
        out = self.ffn_en(out)
        out = self.fc(out).view(batch_size, -1)
        
        if self.args.isinfer:
            out = self.activation(out)
            
        return out
    
class Feed_Forward_block(nn.Module):
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        
        return self.layer2(x)