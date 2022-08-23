import os
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

import torch
from torch import nn
import torch.nn.functional as F
import math

class Simple_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc, num_layers, num_classes):
        super(Simple_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_size_fc = hidden_size_fc

        self.layernorm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(4*hidden_size + 3*hidden_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
        self.fc = nn.Linear(1024,512)
        self.fc1 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, hidden_size_fc)
        self.fc3 = nn.Linear(hidden_size_fc, num_classes)
        self.ft = nn.Linear(1024,768)
        self.fa = nn.Linear(768,1024)

        #drop
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x1, x2, x3, x4, audio):

        #norm 
        #seq_len, batch, feature_dim = x1.size()
        #norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
        #x1 = norm2(x1.transpose(0, 1)).transpose(0, 1)
        #x2 = norm2(x2.transpose(0, 1)).transpose(0, 1)
        text = (x1+x2+x3+x4) / 4
        #audio = self.ft(text)
        text = self.fa(audio)
        text_audio = torch.cat((text, audio),dim = -1)
   
        h0 = torch.zeros(2*self.num_layers, text_audio.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(2*self.num_layers, text_audio.size(1), self.hidden_size).cuda()
        
        out, _ = self.lstm(text_audio, (h0, c0))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        logits =  F.log_softmax(out, 2)

        return out, logits


if __name__ == '__main__':
    #Hyper-parameters
    input_size = 4096
    hidden_size = 256
    hidden_size_fc = 256
    num_layers = 2
    num_classes =7

    x = torch.randn(14, 2, 1024).cuda()
    audio = torch.randn(14, 2, 768).cuda()

    model = Simple_LSTM(input_size, hidden_size, hidden_size_fc, num_layers, num_classes).cuda()
    output, logits = model(x,x,x,x, audio)
    print('output',output, output.shape)
    print('logits',logits, logits.shape)


# class Simple_LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, hidden_size_fc, num_layers, num_classes, input_size_a, hidden_size_a, hidden_size_a2):
#         super(Simple_LSTM, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.hidden_size_fc = hidden_size_fc

#         #input (seq, batch, hidden size)
#         self.layernorm = nn.LayerNorm(input_size)
#         self.lstm1 = nn.LSTM(4*hidden_size + 2*hidden_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
#         #self.lstm2 = nn.LSTM(2*hidden_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
#         self.fc = nn.Linear(input_size, 4*hidden_size)
#         self.fca1 = nn.Linear(input_size_a,hidden_size_a)
#         self.fca2 = nn.Linear(hidden_size_a, hidden_size_a2)
#         self.fc1 = nn.Linear(2*hidden_size, hidden_size_fc)
#         self.fc2 = nn.Linear(hidden_size_fc, num_classes)
#         #drop
#         self.drop = nn.Dropout(p=0.3)

#     def forward(self, x1, x2, x3, x4, audio):

#         #norm 
#         #seq_len, batch, feature_dim = x1.size()
#         #norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
#         #x1 = norm2(x1.transpose(0, 1)).transpose(0, 1)
#         #x2 = norm2(x2.transpose(0, 1)).transpose(0, 1)
        
#         #x = (x1+x2) / 2
#         #x = (x1+x2+x3+x4) / 4
#         x = torch.cat((x1,x2,x3,x4),dim = -1)
#         x  = self.fc(x)
#         audio = F.relu(self.fca1(audio))
#         audio = F.relu(self.fca2(audio))
#         h0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
#         c0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
        
#         #concatenate
#         text_audio = torch.cat((x,audio), dim=-1)
#         #x = self.layernorm(x)
#         #out (seq, batch, hidden size)
#         out, _ = self.lstm1(text_audio, (h0, c0))
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#         logits =  F.log_softmax(out, 2)

#         return out, logits


# if __name__ == '__main__':
#     #Hyper-parameters
#     input_size = 4096
#     hidden_size = 256
#     hidden_size_fc = 256
#     num_layers = 2
#     num_classes =7

#     input_size_a = 6552
#     hidden_size_a = 2048 
#     hidden_size_a2 = 512

#     x = torch.randn(14, 2, 1024).cuda()
#     audio = torch.randn(14, 2, 6552).cuda()

#     model = Simple_LSTM(input_size, hidden_size, hidden_size_fc, num_layers, num_classes, input_size_a, hidden_size_a, hidden_size_a2).cuda()
#     output, logits = model(x,x,x,x, audio)
#     print('output',output, output.shape)
#     print('logits',logits, logits.shape)

#模型3
#fc+lstm+residual
class submodelsresidual(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc, num_layers, num_classes):
        super(submodelsresidual,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_size_fc = hidden_size_fc

        #input (seq, batch, hidden size)
        self.fc1 = nn.Linear(4096, 1024)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
        #self.lstm2 = nn.LSTM(2*hidden_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
        #self.layernorm = nn.LayerNorm(input_size)

        self.fc2 = nn.Linear(2*hidden_size + input_size, hidden_size_fc)
        self.fc3 = nn.Linear(hidden_size_fc, num_classes)
   
        #drop
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x1, x2, x3, x4):

        #norm 
        # seq_len, batch, feature_dim = x1.size()
        # norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
        # x1 = norm2(x1.transpose(0, 1)).transpose(0, 1)
        # x2 = norm2(x2.transpose(0, 1)).transpose(0, 1)
        
        #statisticls
        #x = (x1+x2) / 2
        #x = torch.cat((x1,x2),dim= -1)
        #cat
        x = torch.cat((x1,x2,x3,x4),dim = -1)
        x = self.drop(self.fc1(x))
   
        #print('x',x.shape)
        h0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
        
        #x = self.layernorm(x)
        #out (seq, batch, hidden size)
        out, _ = self.lstm1(x, (h0, c0))
        #out, _ = self.lstm2(out, (h0, c0))
        out = torch.cat((out,x),dim = -1)
        #print('out',out.shape)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        logits =  F.log_softmax(out, 2)

        return out, logits

# if __name__ == '__main__':
#     #Hyper-parameters
#     input_size = 1024
#     hidden_size = 256
#     hidden_size_fc = 128
#     num_layers = 2
#     num_classes =7

#     x = torch.randn(14, 2, 1024).cuda()

#     model = submodelsresidual(input_size, hidden_size, hidden_size_fc, num_layers, num_classes).cuda()
#     output, logits = model(x,x,x,x)
#     print('output',output.shape)
#     print('logits',logits, logits.shape)

class submodelsresiduals(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc, num_layers, num_classes):
        super(submodelsresiduals,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_size_fc = hidden_size_fc

        #input (seq, batch, hidden size)
        self.fc1 = nn.Linear(4096, 1024)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
        #self.lstm2 = nn.LSTM(2*hidden_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
        #self.layernorm = nn.LayerNorm(input_size)

        self.fc2 = nn.Linear(2*hidden_size + input_size, hidden_size_fc)
        self.fc3 = nn.Linear(hidden_size_fc + input_size, hidden_size_fc)
        self.fc4 = nn.Linear(hidden_size_fc, num_classes)
   
        #drop
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x1, x2, x3, x4):

        #norm 
        # seq_len, batch, feature_dim = x1.size()
        # norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
        # x1 = norm2(x1.transpose(0, 1)).transpose(0, 1)
        # x2 = norm2(x2.transpose(0, 1)).transpose(0, 1)
        
        #statisticls
        #x = (x1+x2) / 2
        #x = torch.cat((x1,x2),dim= -1)
        #cat
        x = torch.cat((x1,x2,x3,x4),dim = -1)
        x = self.fc1(x)
   
        #print('x',x.shape)
        h0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
        
        #x = self.layernorm(x)
        #out (seq, batch, hidden size)
        out, _ = self.lstm1(x, (h0, c0))
        #out, _ = self.lstm2(out, (h0, c0))
        out = torch.cat((out,x),dim = -1)
        #print('out',out.shape)
        out = F.relu(self.fc2(out))
        out = torch.cat((out,x),dim = -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        logits =  F.log_softmax(out, 2)

        return out, logits

# if __name__ == '__main__':
#     #Hyper-parameters
#     input_size = 1024
#     hidden_size = 256
#     hidden_size_fc = 128
#     num_layers = 2
#     num_classes =7

#     x = torch.randn(14, 2, 1024).cuda()

#     model = submodelsresiduals(input_size, hidden_size, hidden_size_fc, num_layers, num_classes).cuda()
#     output, logits = model(x,x,x,x)
#     print('output',output.shape)
#     print('logits',logits, logits.shape)

# class submodelsresidual(nn.Module):
#     def __init__(self, input_size, hidden_size, hidden_size_fc, num_layers, num_classes):
#         super(submodelsresidual,self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.hidden_size_fc = hidden_size_fc

#         #input (seq, batch, hidden size)
#         self.fc1 = nn.Linear(4096, 1024)
#         self.fc2 = nn.Linear(1024,256)

#         self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first= False, bidirectional=True)
#         self.lstm2 = nn.LSTM(hidden_size, hidden_size_fc, num_layers, batch_first= False, bidirectional=True)
#         #self.layernorm = nn.LayerNorm(input_size)

#         self.fc3 = nn.Linear(2*hidden_size + input_size + 2*hidden_size_fc, 2*hidden_size + input_size + 2*hidden_size_fc)
#         self.fc4 = nn.Linear(2*hidden_size + input_size + 2*hidden_size_fc, hidden_size_fc)
#         self.fc5 = nn.Linear(hidden_size_fc, num_classes)
   
#         #drop
#         self.drop = nn.Dropout(p=0.3)

#     def forward(self, x1, x2, x3, x4):

#         #norm 
#         # seq_len, batch, feature_dim = x1.size()
#         # norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
#         # x1 = norm2(x1.transpose(0, 1)).transpose(0, 1)
#         # x2 = norm2(x2.transpose(0, 1)).transpose(0, 1)
        
#         #statisticls
#         #x = (x1+x2) / 2
#         #x = torch.cat((x1,x2),dim= -1)
#         #cat
#         x = torch.cat((x1,x2,x3,x4),dim = -1)
#         #avg
#         x_avg = (x1+x2+x3+x4)/4
#         #x = (x1+x2+x3+x4) / 4
#         #x = x1+x2+x3+x4
#         #Set initial hidden states (and cell states for LSTM)
#         x = self.fc1(x)
#         x_avg = self.fc2(x_avg)
   
#         #print('x',x.shape)
#         h0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()
#         c0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda()

#         ha= torch.zeros(2*self.num_layers, x.size(1), self. hidden_size_fc).cuda()
#         ca = torch.zeros(2*self.num_layers, x.size(1), self. hidden_size_fc).cuda()
        
#         #x = self.layernorm(x)
#         #out (seq, batch, hidden size)
#         out, _ = self.lstm1(x, (h0, c0))
#         out_avg, avg_ = self.lstm2(x_avg,(ha, ca))
#         #out, _ = self.lstm2(out, (h0, c0))
#         out = torch.cat((out,x,out_avg),dim = -1)
#         #print('out',out.shape)
#         out = F.relu(self.fc3(out))
#         out = F.relu(self.fc4(out))
#         out = self.fc5(out)
#         logits =  F.log_softmax(out, 2)

#         return out, logits

# if __name__ == '__main__':
#     #Hyper-parameters
#     input_size = 1024
#     hidden_size = 256
#     hidden_size_fc = 128
#     num_layers = 2
#     num_classes =7

#     x = torch.randn(14, 2, 1024).cuda()

#     model = submodelsresidual(input_size, hidden_size, hidden_size_fc, num_layers, num_classes).cuda()
#     output, logits = model(x,x,x,x)
#     print('output',output.shape)
#     print('logits',logits, logits.shape)

#模型2
#self-attention
class selfattention(nn.Module):
    def __init__(self, key_size, num_hiddens, dropout_c, **kwargs):
        super(selfattention, self).__init__(**kwargs)

        self.key_size = key_size
        self.num_hiddens = num_hiddens
        self.dropout = dropout_c
        self.dropout = nn.Dropout(dropout_c)
        self.W_q = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=False)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attn_weights = nn.functional.softmax(scores, dim=-1)
        print('attn_weights', attn_weights, attn_weights.shape)
        return torch.bmm(self.dropout(attn_weights), values), attn_weights

# if __name__ == '__main__':

#    X = torch.normal(0, 1, (2, 9, 1024))
#    print('X',X.shape)
#    # hyparameters
#    key_size = 1024
#    num_hiddens = 1024
#    dropout = 0.3
#    model = selfattention(key_size, num_hiddens, dropout)
#    print(model)
#    output, attention_weights = model(X,X,X)
#    print('attention_weights shape', attention_weights, attention_weights.shape)
#    print('output shape', output.shape)

class selfattentiontext(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc,  hidden_size_fc2, num_layers, num_classes):
        super(selfattentiontext,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_fc = hidden_size_fc
        self.hidden_size_fc2 = hidden_size_fc2
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.3)

        #self.layernorm = nn.LayerNorm(input_size) 
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(4096,1024)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= False, bidirectional=True)

        #attention
        self.attention = selfattention(2*hidden_size, 2*hidden_size, 0.3)

        self.fc2 = nn.Linear(2*hidden_size + input_size, hidden_size_fc)
        self.fc3 = nn.Linear(hidden_size_fc, hidden_size_fc2)
        self.fc4 = nn.Linear(hidden_size_fc2, num_classes)

    def forward(self, x1, x2, x3, x4):

        text = torch.cat((x1,x2,x3,x4),dim = -1)
        #print('text',text.shape)
        #text = self.layernorm(text)
        text = self.fc1(text)

        h0 = torch.zeros(2*self.num_layers, text.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(2*self.num_layers, text.size(1), self.hidden_size).cuda()
        
        #out (batch, sequence, hidden_size)
        out, _ = self.lstm(text, (h0, c0))
        #attention
        out_ = out.transpose(1,0)
        attention, _ = self.attention(out_,out_,out_)
        attention_ = attention.transpose(1,0)
        #print('attention',attention.shape)
        #combination
        out = torch.cat((attention_, text), dim = -1)

    
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        logits =  F.log_softmax(out, 2)

        #out (seq, batch, hidden size)
        #logits (seq, batch, hidden size)


        return out, logits


# if __name__ == '__main__':

#     #Hyper-parameters 
#     input_size = 1024
#     hidden_size = 256
#     hidden_size_fc = 512
#     hidden_size_fc2 = 64
#     num_layers = 2
#     num_classes =7

#     x = torch.randn(9, 2, 1024).cuda()

#     model = selfattentiontext(input_size,hidden_size,hidden_size_fc,hidden_size_fc2,num_layers, num_classes).cuda()
#     out, logits = model(x,x,x,x)
#     print('output', logits, logits.shape)


class selfattentiontext(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc,  hidden_size_fc2, num_layers, num_classes):
        super(selfattentiontext,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_fc = hidden_size_fc
        self.hidden_size_fc2 = hidden_size_fc2
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.3)

        self.layernorm = nn.LayerNorm(input_size) 
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True, bidirectional=True)

        #attention
        self.attention = selfattention(4096, 4096, 0.3)

        self.fc1 = nn.Linear(2*hidden_size, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, hidden_size_fc2)
        self.fc3 = nn.Linear(hidden_size_fc2, num_classes)

    def forward(self, x1, x2, x3, x4):

        #x (seq, batch, feature)
        text = torch.cat((x1,x2,x3,x4),dim = -1)
        #print('text',text.shape)
        #text = self.layernorm(text)

        #set initial hidden states (and cell states for LSTM)
        #attention
        attention, _ = self.attention(text,text,text)
        #print('attention',attention.shape)
       
        #text + selfattention
        text_attention = torch.cat((attention, text), dim = -1)

        h0 = torch.zeros(2*self.num_layers, text_attention.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(2*self.num_layers, text_attention.size(0), self.hidden_size).cuda()
        
        #out (batch, sequence, hidden_size)
        out, _ = self.lstm(text_attention, (h0, c0))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        logits =  F.log_softmax(out, 2)

        #out (seq, batch, hidden size)
        #logits (seq, batch, hidden size)


        return out, logits


# if __name__ == '__main__':

#     #Hyper-parameters 
#     input_size = 8192
#     hidden_size = 1024
#     hidden_size_fc = 256
#     hidden_size_fc2 = 64
#     num_layers = 2
#     num_classes =7

#     x = torch.randn(2, 9, 1024).cuda()

#     model = selfattentiontext(input_size,hidden_size,hidden_size_fc,hidden_size_fc2,num_layers, num_classes).cuda()
#     out, logits = model(x,x,x,x)
#     print('output', logits, logits.shape)

# #模型3
#self attenton +  cross attention

class FFN(nn.Module):
    def __init__(self, ffn_input_size, ffn_hidden_size, ffn_hidden_size_out, **kwargs):
       
        super(FFN, self).__init__(**kwargs)
        self.ffn_1 = nn.Linear(ffn_input_size, ffn_hidden_size)
        self.ffn_2 = nn.Linear(ffn_hidden_size, ffn_hidden_size_out)
        
        
    def forward(self, X):
        output = self.ffn_2(F.relu(self.ffn_1(X)))
        return output

class crossattention(nn.Module):
    def __init__(self, key_size, num_hiddens, dropout_c, **kwargs):
        super(crossattention, self).__init__(**kwargs)

        self.key_size = key_size
        self.num_hiddens = num_hiddens
        self.dropout = dropout_c
        self.dropout = nn.Dropout(dropout_c)
        self.W_q = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=False)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attn_weights = nn.functional.softmax(scores, dim=-1)
        #print('attn_weights', attn_weights.shape,attn_weights)

        return torch.bmm(self.dropout(attn_weights), values), attn_weights

# if __name__ == '__main__':

#    X = torch.normal(0, 1, (2, 6, 6552))
#    Y = torch.normal(0, 1, (2, 6, 1024))
 
#    #hyparameters 
#    key_size = 6552
#    num_hiddens = 1024
#    dropout = 0.3
#    model = crossattention(key_size, num_hiddens, dropout)
#    output, attention_weights = model(X,Y,Y)
#    print('attention_weights shape', attention_weights, attention_weights.shape)
#    print('output', output, output.shape)

#audio + text + cossattention + selfattention 
class AttentionBimodal(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc,  hidden_size_fc2, num_layers, num_classes):
        super(AttentionBimodal,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_fc = hidden_size_fc
        self.hidden_size_fc2 = hidden_size_fc2
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.3)

        self.layernorm = nn.LayerNorm(input_size) 
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True, bidirectional=True)

        #crossattention
        self.audio_text_attention = crossattention(6552, 1024, 0.2)
        self.text_audio_attention = crossattention(1024, 6552,0.2)

        #embedding
        self.audio_embedding = FFN(6552, 1024, 300)
        self.text_embedding = FFN(1024, 1024, 300)


        self.fc1 = nn.Linear(2*hidden_size, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, hidden_size_fc2)
        self.fc3 = nn.Linear(hidden_size_fc2, num_classes)

    def forward(self, audio, text):

        #x = self.layernorm(x)
        #set initial hidden states (and cell states for LSTM)
        #(seq,batch,feature)
        audio = audio.transpose(1,0)
        text = text.transpose(1,0)

        #crossattention
        audio_text_attention, _ = self.audio_text_attention(audio,text,text)
        text_audio_attention, _ = self.text_audio_attention(text,audio,audio)
        audio_attembedding = self.audio_embedding(text_audio_attention)
        text_attembedding = self.text_embedding(audio_text_attention)

        #embedding
        audio_embedding = self.audio_embedding(audio)
        text_embedding = self.text_embedding(text)

        #attention + embedding
        audio_text = torch.cat((text_embedding, audio_embedding, audio_attembedding, text_attembedding), dim = -1)

        h0 = torch.zeros(2*self.num_layers, audio_text.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(2*self.num_layers, audio_text.size(0), self.hidden_size).cuda()
        
        #out (batch, sequence, hidden_size)
        out, _ = self.lstm(audio_text, (h0, c0))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        logits =  F.log_softmax(out, 2)

        #out (seq, batch, hidden size)
        #logits (seq, batch, hidden size)
        out = out.transpose(1,0)
        logits = logits.transpose(1,0)
        return out, logits


# if __name__ == '__main__':

#     #Hyper-parameters 
#     input_size = 1200
#     hidden_size = 256
#     hidden_size_fc = 256
#     hidden_size_fc2 = 64
#     num_layers = 2
#     num_classes =7

#     audio = torch.randn(9, 2, 6552).cuda()
#     text = torch.randn(9, 2, 1024).cuda()

#     model = AttentionBimodal(input_size,hidden_size,hidden_size_fc,hidden_size_fc2,num_layers, num_classes).cuda()
#     output, logits = model(audio,text)
#     print('output', output, output.shape) 
#     print('logts', logits, logits.shape)

#模型4 tensorfusion
#rnn_dialogue中tensorfusion具体实施

