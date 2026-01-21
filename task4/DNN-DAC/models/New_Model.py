# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding='random'):
        self.model_name = 'New_Model'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 16                                            # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切) - slightly reduced for RCNN memory
        self.learning_rate = 2e-5                                       # 学习率
        
        self.bert_path = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        
        self.hidden_size = 768
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout = 0.1 # RoBERTa default

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True # Fine-tune BERT
            
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, 
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]

        encoder_out, text_cls = self.bert(context, attention_mask=mask, return_dict=False)
        
        # LSTM
        lstm_out, _ = self.lstm(encoder_out)
        
        # Combined Features:
        # 1. Global Max Pooling on LSTM output (captures strongest signals in sequence)
        out = lstm_out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        
        # 2. Concat with BERT CLS token (global context)
        out = torch.cat((out, text_cls), 1)
        
        out = self.fc(out)
        return out
