# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import os
class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding='random'):
        self.model_name = 'New_Model'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BERT模型选择（中文任务推荐）
        self.bert_path = 'bert-base-chinese'  # 基础中文BERT
        # 备选：'hfl/chinese-bert-wwm'（全词掩码，效果更好但下载慢）
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        
        # 训练参数（针对BERT微调优化）
        self.dropout = 0.3
        self.require_improvement = 200  # 早停耐心值
        self.num_classes = len(self.class_list)
        self.n_vocab = 0  # BERT使用自己的词表
        
        # BERT微调超参数
        self.num_epochs = 4            # BERT微调通常3-5轮足够
        self.batch_size = 16           # 根据GPU内存调整
        self.pad_size = 128            # 最大序列长度
        self.learning_rate = 2e-5      # BERT需要非常小的学习率
        self.weight_decay = 0.01       # 权重衰减
        self.gradient_accumulation_steps = 2  # 梯度累积步数
        
        # 嵌入层参数（BERT固定为768）
        self.embed = 768


class Model(nn.Module):
    """BERT文本分类模型"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        print(f"[DEBUG] 配置中的 bert_path: {config.bert_path}")
        print(f"[DEBUG] 该路径是否存在: {os.path.exists(config.bert_path)}")
        # 加载BERT配置
        bert_config = BertConfig.from_pretrained(
            config.bert_path,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            output_hidden_states=True  # 输出所有隐藏状态（可选）
        )
        
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(
            config.bert_path,
            config=bert_config
        )
        
        # 冻结BERT前几层（可选，加速训练）
        # for param in list(self.bert.parameters())[:100]:
        #     param.requires_grad = False
        
        # 分类器
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(bert_config.hidden_size, config.num_classes)
        
        # 初始化分类层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
        # 层归一化（可选）
        self.layer_norm = nn.LayerNorm(bert_config.hidden_size)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 元组 (input_ids, seq_len)
                为了兼容原有框架
        Returns:
            logits: 分类输出
        """
        # 解包输入
        input_ids = x[0]  # [batch_size, seq_len]
        
        # 生成attention mask（非padding位置为1）
        attention_mask = (input_ids != 0).long()
        
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS]位置的隐藏状态作为句子表示
        # pooler_output已经经过了非线性变换和池化
        pooled_output = outputs.pooler_output
        
        # 层归一化
        pooled_output = self.layer_norm(pooled_output)
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.fc(pooled_output)
        
        return logits


# 增强版BERT（可选）
class EnhancedBERT(nn.Module):
    """增强版BERT模型（结合CNN特征）"""
    def __init__(self, config):
        super(EnhancedBERT, self).__init__()
        self.config = config
        
        # 加载BERT
        bert_config = BertConfig.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)
        
        # CNN层提取局部特征
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 128
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, bert_config.hidden_size))
            for k in self.filter_sizes
        ])
        
        # 特征融合
        bert_dim = bert_config.hidden_size
        cnn_dim = self.num_filters * len(self.filter_sizes)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_dim + cnn_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(256, config.num_classes)
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def conv_and_pool(self, x, conv):
        """卷积+池化"""
        x = nn.functional.relu(conv(x)).squeeze(3)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        input_ids = x[0]
        attention_mask = (input_ids != 0).long()
        
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # BERT特征
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls_output = outputs.pooler_output  # [batch, hidden]
        
        # CNN特征
        cnn_input = sequence_output.unsqueeze(1)  # [batch, 1, seq_len, hidden]
        cnn_features = [self.conv_and_pool(cnn_input, conv) for conv in self.convs]
        cnn_concat = torch.cat(cnn_features, dim=1)
        
        # 特征融合
        combined = torch.cat([cls_output, cnn_concat], dim=1)
        combined = self.dropout(combined)
        
        # 分类
        logits = self.feature_fusion(combined)
        
        return logits