import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------
# 1. toy graph 定義
# -------------------------
edge_index = torch.tensor([
    [0,1,1,2,1,3,0,4,4,5],
    [1,0,2,1,3,1,4,0,5,4]
], dtype=torch.long)

num_nodes = 6
T = 10  # 10 歷史時間步

# -------------------------
# 2. toy data 生成
# -------------------------
torch.manual_seed(0)

# 流量 + 天氣 + 事件
flows = torch.randn(T, num_nodes, 1) * 10 + 50
weather = torch.randint(0, 2, (T, num_nodes, 1)).float()
incident = torch.randint(0, 2, (T, num_nodes, 1)).float()

x_seq = torch.cat([flows, weather, incident], dim=2)  # (T, N, 3)

# label: 下一步 flow
next_flow = flows[-1] * 0.7 + weather[-1] * 5 + incident[-1] * 10

# -------------------------
# 3. LSTM-GNN 模型
# -------------------------
class LSTM_GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = GCNConv(3, 16)
        self.gnn2 = GCNConv(16, 8)
        self.lstm = torch.nn.LSTM(8, 16, batch_first=True)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, x_seq, edge_index):
        gnn_outputs = []
        for t in range(x_seq.size(0)):
            x = x_seq[t]
            x = F.relu(self.gnn1(x, edge_index))
            x = F.relu(self.gnn2(x, edge_index))
            gnn_outputs.append(x.unsqueeze(0))  # (1, N, 8)

        gnn_outputs = torch.cat(gnn_outputs, dim=0)  # (T, N, 8)
        gnn_outputs = gnn_outputs.transpose(0,1)     # (N, T, 8)

        lstm_out, _ = self.lstm(gnn_outputs)         # (N, T, 16)
        pred = self.fc(lstm_out[:, -1])              # (N, 1)
        return pred

model = LSTM_GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# -------------------------
# 4. 訓練 300 epochs（並記錄 loss）
# -------------------------
losses = []
num_epochs = 300
for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred = model(x_seq, edge_index)
    loss = loss_fn(pred, next_flow)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    # 每 10 個 epoch 印一次進度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} - loss: {loss.item():.6f}")

# -------------------------
# 5. 結果
# -------------------------
print("真實下一步流量:", next_flow.squeeze())
print("模型預測流量:", pred.detach().squeeze())

# -------------------------
# 6. 繪製並儲存 loss vs epoch（若有記錄）
# -------------------------
if len(losses) > 0:
    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(8,4))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    out_path = 'training_loss_cc.png'
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss plot to: {out_path}")
    try:
        plt.show()
    except Exception:
        pass