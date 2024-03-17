import torch
import torch.nn.functional as F

from models import GPModel, MultilayerPerceptron


class IntegratedModel(torch.nn.Module):
    def __init__(self, args):
        super(IntegratedModel, self).__init__()
        self.gp_model = GPModel(args)  # 实例化 GPModel
        self.mlp = MultilayerPerceptron(args)  # 实例化 MultilayerPerceptron

    def forward(self, data):
        # 获取GPModel的输出特征
        gp_output = self.gp_model(data)

        # 将GPModel的输出特征作为MLP的输入
        mlp_output, mlp_features = self.mlp(gp_output)

        return mlp_output, mlp_features

# # 训练模型时使用的伪代码
# integrated_model = IntegratedModel(args)  # 实例化整合模型
# optimizer = torch.optim.Adam(integrated_model.parameters(), lr=0.01)  # 选择优化器
#
# for epoch in range(num_epochs):
#     for data in data_loader:  # 假设 data_loader 是你的数据迭代器
#         optimizer.zero_grad()  # 清零梯度
#         output, features = integrated_model(data)  # 前向传播
#         # 计算损失，这里需要你提供真实的标签 data.y
#         loss = F.mse_loss(output, data.y)  # 假设是回归任务
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
