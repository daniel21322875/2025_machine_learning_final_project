import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------
# 1. 生成 toy traffic data
# -----------------------
np.random.seed(0)
N = 2000

# 車流量（上一分鐘）
flow = np.random.normal(50, 15, N)  # 每分鐘車數

# 天氣（0:晴天, 1:雨天）
weather = np.random.choice([0, 1], N, p=[0.7, 0.3])

# 時段（0:早, 1:午, 2:晚, 3:尖峰）
time = np.random.choice([0, 1, 2, 3], N, p=[0.3,0.3,0.2,0.2])

# 事件（0:正常, 1:事故）
incident = np.random.choice([0, 1], N, p=[0.9, 0.1])

# 真實模型（toy）：雨天 + 尖峰 + 事故 = 車流暴增或暴減
next_flow = (
    0.6 * flow +
    10 * weather +
    15 * (time == 3) +
    20 * incident +
    np.random.normal(0, 5, N)
)

# 正規化
X = np.column_stack([flow, weather, time, incident])
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = next_flow / next_flow.max()

# -----------------------
# 2. 神經網路（MLP）
# -----------------------
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# -----------------------
# 3. 訓練
# -----------------------
history = model.fit(X, y, epochs=25, batch_size=32)

# -----------------------
# 4. 繪製訓練過程 loss vs epoch
# -----------------------
if hasattr(history, 'history') and 'loss' in history.history:
    losses = history.history['loss']
    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    out_path = 'training_loss_final213.png'
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss plot to: {out_path}")
    try:
        plt.show()
    except Exception:
        pass

# -----------------------
# 4. 預測示範
# -----------------------
sample = np.array([[60, 1, 3, 0]]) # flow, weather, time, incident
sample = (sample - X.mean(axis=0)) / X.std(axis=0)

print("預測下一分鐘車流量 (normalized):", model.predict(sample)[0][0])
