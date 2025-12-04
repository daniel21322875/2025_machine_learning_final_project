import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------
# 1. 建 toy graph
# ---------------------
edge_index = torch.tensor([
    [0,1,1,2,1,3,0,4,4,5],   # source
    [1,0,2,1,3,1,4,0,5,4]    # target
], dtype=torch.long)

num_nodes = 6

# 生成 toy 特徵（flow, weather, incident）
torch.manual_seed(0)
flow = torch.randn(num_nodes, 1) * 10 + 50
weather = torch.randint(0, 2, (num_nodes, 1)).float()
incident = torch.randint(0, 2, (num_nodes, 1)).float()

x = torch.cat([flow, weather, incident], dim=1)

# 生成 toy label：下一分鐘 flow
y = flow * 0.7 + weather * 5 + incident * 10 + torch.randn_like(flow)*2

data = Data(x=x, edge_index=edge_index, y=y)

# ---------------------
# 2. 定義簡易 GNN
# ---------------------
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# ---------------------
# 3. 訓練 300 epochs
# ---------------------
losses = []
for epoch in range(300):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()

    # save loss and print progress every 10 epochs
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} - loss: {loss.item():.4f}")

# 印出最終訓練 loss（以訓練後模型計算）
final_loss = loss_fn(model(data.x, data.edge_index), data.y).item()
print(f"Final training loss: {final_loss:.4f}")

# ---------------------
# 4. 預測
# ---------------------
pred = model(data.x, data.edge_index).detach()
print("真實下一分鐘:", data.y.squeeze())
print("GNN 預測值:", pred.squeeze())

# ---------------------
# 5. 繪製 loss vs epoch
# ---------------------
if len(losses) > 0:
    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(8,4))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    out_path = 'training_loss.png'
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss plot to: {out_path}")
    try:
        plt.show()
    except Exception:
        # In headless environments plt.show() may fail; that's fine because we saved the file.
        pass