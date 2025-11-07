"""
Model definitions for FSDP training demo
"""

import torch.nn as nn


class SimpleLinearModel(nn.Module):
    """简单的4层Linear模型，用于FSDP研究"""
    def __init__(self, input_size=512, hidden_size=512, output_size=10000):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 4层Linear层
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        
        # 添加激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size) 或 (batch_size, input_size)
        if x.dim() == 2:
            # 如果是2D，直接处理
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            x = self.relu(x)
            x = self.layer4(x)
        else:
            # 如果是3D (batch_size, seq_len, input_size)，对每个位置应用
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, self.input_size)  # (batch_size * seq_len, input_size)
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            x = self.relu(x)
            x = self.layer4(x)
            x = x.view(batch_size, seq_len, self.output_size)  # (batch_size, seq_len, output_size)
        return x
