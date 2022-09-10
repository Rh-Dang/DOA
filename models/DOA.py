from __future__ import division
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput

import scipy.sparse as sp
import numpy as np
import scipy.io as scio

class Multihead_Attention(nn.Module):    
    """
    multihead_attention
    """

    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 num_heads=1,                  
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)   
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_out = nn.Linear(num_heads, 1)

    def forward(self,
                Q, K):
        """
        :param Q: A 3d tensor with shape of [T_q, C_q]   
        :param K: A 3d tensor with shape of [T_k, C_k]   
        :param V: A 3d tensor with shape of [T_v, C_v]   
        :return:
        """
        num_heads = self.num_heads
        N = 1                                           #batch
        Q = Q.unsqueeze(dim = 0)             
        K = K.unsqueeze(dim = 0)

        # Linear projections
        Q_l = nn.ReLU()(self.linear_Q(Q))                         
        K_l = nn.ReLU()(self.linear_K(K))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)  
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)                    
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))    #(h*N, T_q(1), T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)   

        # Dropouts   
        outputs = self.dropout(outputs) 
        outputs = outputs.split(N, dim=0)
        outputs = torch.cat(outputs, dim=1)  #(1, num_heads, num_point)
        outputs = outputs.transpose(1,2)     #(1, num_point, num_heads)
        outputs = self.linear_out(outputs)   ##(1, num_point, 1)
        outputs = nn.Softmax(dim=1)(outputs).squeeze(dim=0)

        # # Residual connection
        # outputs = outputs + Q_l

        # # Normalize
        # outputs = self.norm(outputs)  # (N, T_q, C)

        return outputs   


class DOA(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(DOA, self).__init__()

        self.image_size = 300
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)

        self.action_at_a = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), requires_grad=False)
        self.action_at_b = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), requires_grad=False)
        self.action_at_scale = nn.Parameter(torch.tensor(0.58), requires_grad=False) 

        self.graph_detection_feature = nn.Sequential(
            nn.Linear(262, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.embed_action = nn.Linear(action_space, 10)

        pointwise_in_channels = 64 + self.num_cate + 10

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)
        self.dropout_rate = 0.35
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.info_embedding = nn.Linear(5,49)
        # self.scene_embedding = nn.Conv2d(86,64,1,1)
        # self.scene_classifier = nn.Linear(64*7*7,4)

        # last layer of resnet18.
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[-2:]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False

        self.W0 = nn.Linear(22, 22, bias=False)
        
     
        self.target_object_attention = torch.nn.Parameter(torch.FloatTensor(self.num_cate, self.num_cate), requires_grad=True)
        self.target_object_attention.data.fill_(1/22)  

     
        self.scene_object_attention = torch.nn.Parameter(torch.FloatTensor(4, self.num_cate, self.num_cate), requires_grad=True)
        self.scene_object_attention.data.fill_(1/22)   

     
        self.attention_weight = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=True)
        self.attention_weight.data.fill_(1/2)


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))    #(64,7,7) -> (64,1,1)
        # self.image_to_attent = nn.Linear(64,22)

        self.muti_head_attention = Multihead_Attention(hidden_dim = 512, C_q = resnet_embedding_sz + 64, C_k = 262, num_heads = 8, dropout_rate = 0.3)
        self.conf_threshod = 0.6
 

        self.num_cate_embed = nn.Sequential(
            nn.Linear(self.num_cate, 32), 
            nn.ReLU(),
            nn.Linear(32, 64),  
            nn.ReLU(),
        )

    def one_hot(self, spa):  

        y = torch.arange(spa).unsqueeze(-1)  
        y_onehot = torch.FloatTensor(spa, spa)  

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)    

        return y_onehot   ## (22,22)

    def embedding(self, state, target, action_embedding_input , target_object):

      
        at_v = torch.mul(target['scores'].unsqueeze(dim=1), target['indicator']) 
        at = torch.mul(torch.max(at_v), self.action_at_scale)  
        action_at = torch.mul(at, self.action_at_a) + self.action_at_b  
  

        target_object = target['indicator']                   
        
        
        action_embedding = F.relu(self.embed_action(action_embedding_input)) 
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)  

        image_embedding = F.relu(self.conv1(state))  

        x = self.dropout(image_embedding)

        target_appear = target['features']
        target_conf = target['scores'].unsqueeze(dim=1)
        target_bbox = target['bboxes'] / self.image_size

        target = torch.cat((target_appear, target_bbox, target_conf, target_object), dim=1)  

        target_object_attention = F.softmax(self.target_object_attention, 0)        

        attention_weight = F.softmax(self.attention_weight, 0)   

        object_attention = target_object_attention * attention_weight[0]
        
        object_select = torch.sign(target_conf - 0.6)  #(22,1)
        object_select[object_select > 0] = 0                       
        object_select[object_select < 0] = - object_select[object_select < 0]   #(1,22)       
        object_select_appear = object_select.squeeze().expand(262, 22).bool()          
        target_mutiHead = target.masked_fill(object_select_appear.t(),0)          

        image_object_attention = self.avgpool(state).squeeze(dim = 2).squeeze(dim = 0).t()   # (1,512)  
        spa = self.one_hot(self.num_cate).to(target.device)                                  # (22,22)
        num_cate_index = torch.mm(spa.t(), target_object).t()
        num_cate_index = self.num_cate_embed(num_cate_index)                                  #（22,64）
        image_object_attention = torch.cat((image_object_attention, num_cate_index), dim = 1)  #(1,512+64=576)
        image_object_attention = self.muti_head_attention(image_object_attention, target_mutiHead)
        

        target_attention= torch.mm(object_attention, target_object)  
        target_attention = target_attention + image_object_attention * attention_weight[1]  

        target = F.relu(self.graph_detection_feature(target))    # 518-128-49  N*49
        target = target * target_attention                        
        target_embedding = target.reshape(1, self.num_cate, 7, 7)    # 1*N*7*7
        target_embedding = self.dropout(target_embedding)  
        ##############################################################################################################################


        x = torch.cat((x, target_embedding, action_reshaped), dim=1)

        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)    


        return out, image_embedding, action_at

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])      #1*1*(64*7*7)
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))  
        
        x = output.reshape([1, self.hidden_state_sz])    #1*512

        actor_out = self.actor_linear(x)   #512 - 6
        critic_out = self.critic_linear_1(x)   #512-64 
        critic_out = self.critic_linear_2(critic_out)   #64-1

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):                                             
        target_object = model_input.target_object  

        state = model_input.state  
        (hx, cx) = model_input.hidden   

        target = model_input.target_class_embedding  
        action_probs = model_input.action_probs     

        x, image_embedding , action_at= self.embedding(state, target, action_probs, target_object)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)
        actor_out = torch.mul(actor_out, action_at)   
        return ModelOutput(
            value=critic_out,            
            logit=actor_out,            
            hidden=(hx, cx),             
            embedding=image_embedding,   
        )
