# Deep Reinforcement Learning cho Maze Crawler — Refined Implementation Plan

## Tóm tắt thay đổi so với bản gốc

Bản plan này là phiên bản refined hoàn chỉnh từ [DRL_README.md](file:///f:/Study/cloud/maze-crawler/DRL_README.md) gốc, khắc phục **tất cả 7 vấn đề nghiêm trọng** đã được chỉ ra trong review:

| # | Vấn đề gốc | Giải pháp trong bản mới |
|:--|:-----------|:----------------------|
| 1 | Action space chỉ 9 actions | Liệt kê đầy đủ 25+ actions → gộp thành 13 unified actions với type-specific masking |
| 2 | Multi-agent variable count chưa giải quyết | Decentralized IPPO + Parameter Sharing, mỗi robot là 1 agent độc lập |
| 3 | Observation height cố định 20 sai | Egocentric local view 11×11 quanh mỗi robot |
| 4 | Thiếu fog-of-war encoding | Thêm kênh visibility mask riêng biệt |
| 5 | Reward shaping chưa đủ | Bổ sung 8 loại reward mới (mine income, combat, tempo, v.v.) |
| 6 | Timeline quá lạc quan | Điều chỉnh thành 6-8 tuần thực tế với milestone gates |
| 7 | Thiếu submission pipeline | Chi tiết ONNX export + CPU inference + tar.gz bundling |

---

## User Review Required

> [!IMPORTANT]
> **Chọn framework huấn luyện**: Plan này đề xuất dùng **sb3-contrib MaskablePPO** (đơn giản hơn) hoặc **PufferLib + CleanRL** (nhanh hơn, nhưng phức tạp setup). Bạn prefer framework nào? T muốn tự xây và thử nghiệm với các agent như COMA, MADDPG, MAPPO, IPPO. Tbh T thích tự build một PPO cơ bản vì có thể tùy biến tốt và hiểu hơn về cách cài đặt.

> [!IMPORTANT]
> **Hardware**: Training DRL cần GPU. Bạn có GPU nào (RTX 3060/4060/4090?) hay dùng cloud (Colab, Kaggle GPU, vast.ai)? Kaggle GPU

> [!WARNING]
> **Meta hiện tại**: Theo community discussion trên Kaggle, meta hiện tại nghiêng về **"Factory Solo + Alpha"** (factory di chuyển liên tục, ít build unit). DRL agent nên được thiết kế để cũng học được chiến thuật minimalist này, không bị bias vào việc build quá nhiều unit. ok

---

## 1. Kiến trúc Tổng Quan

### Phương pháp: Decentralized IPPO + Parameter Sharing

```
┌──────────────────────────────────────────────────────────────┐
│                   Kaggle Environment (Crawl)                  │
│  obs.robots = {"uid1": [...], "uid2": [...], ...}            │
└───────────────┬──────────────────────────────▲────────────────┘
                │ Raw Observation               │ actions = {"uid1": "NORTH", ...}
                ▼                               │
┌──────────────────────────────────────────────────────────────┐
│              CrawlGymnasiumWrapper (env_wrapper.py)           │
│                                                               │
│  Cho MỖI robot của phe mình:                                 │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ 1. Trích xuất local view 11×11 (12 kênh) quanh robot│     │
│  │ 2. Tạo scalar features (energy, type, cooldowns)     │     │
│  │ 3. Sinh action mask (13 actions, lọc theo type+wall)│     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  Output: List[(obs_tensor, scalar, mask)] cho N robots       │
└───────────────┬──────────────────────────────▲────────────────┘
                │ Per-robot (obs, mask)         │ Per-robot action_id
                ▼                               │
┌──────────────────────────────────────────────────────────────┐
│          Shared Policy Network (CNN-GRU Actor-Critic)         │
│                                                               │
│  Input: local_map (12, 11, 11) + scalars (K,)               │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│     │ Conv2D×3 │──▶│ GRU Cell │──▶│ FC Heads │              │
│     └──────────┘   └──────────┘   └──────────┘              │
│                                    ├─▶ Actor (13) + mask     │
│                                    └─▶ Critic (1)            │
│                                                               │
│  MỌI robot chia sẻ chung weights (Parameter Sharing)         │
└──────────────────────────────────────────────────────────────┘
```

**Tại sao Decentralized thay vì Centralized:**
- Centralized Action Map `20×20×A` có action space quá lớn (~10,000) → sample efficiency cực thấp
- Số lượng robot thay đổi mỗi turn → Centralized phải pad/mask liên tục, rất phức tạp
- Decentralized + Parameter Sharing: mỗi robot chỉ cần 1 forward pass qua cùng 1 network → scale tự nhiên với N robots bất kỳ
- Cách tiếp cận này đã được chứng minh hiệu quả trong Lux AI, MicroRTS, và các game tương tự

---

## 2. Thiết kế Chi Tiết Observation Space

### 2.1. Local Egocentric Map — `(12, 11, 11)` per robot

Mỗi robot nhận một **local view 11×11 ô** xung quanh vị trí hiện tại (radius = 5, vừa bằng tầm nhìn Scout). Nếu ô nằm ngoài map hoặc trong fog → giá trị = 0.

| Kênh | Tên | Giá trị | Mô tả |
|:----:|:----|:--------|:------|
| 0 | `wall_north` | 0/1 | Có tường phía Bắc tại ô đó |
| 1 | `wall_east` | 0/1 | Có tường phía Đông |
| 2 | `wall_south` | 0/1 | Có tường phía Nam |
| 3 | `wall_west` | 0/1 | Có tường phía Tây |
| 4 | `visibility` | 0/1 | **1 = đã khám phá, 0 = fog chưa biết** (giải quyết vấn đề fog-of-war) |
| 5 | `my_robots` | 0-4 | Loại robot phe mình: 0=trống, 1=Factory, 2=Scout, 3=Worker, 4=Miner |
| 6 | `my_robot_energy` | 0.0-1.0 | Năng lượng robot phe mình (chuẩn hóa bởi maxEnergy của type đó) |
| 7 | `enemy_robots` | 0-4 | Loại robot đối phương nhìn thấy (0 = trống/fog) |
| 8 | `crystals` | 0.0-1.0 | Năng lượng crystal (chuẩn hóa / 50) |
| 9 | `mining_nodes` | 0/1 | Có mining node tại ô đó |
| 10 | `mines` | -1/0/+1 | Mine: +1 = phe mình, -1 = đối phương, 0 = không có |
| 11 | `scroll_danger` | 0.0-1.0 | `1.0 - (row - southBound) / viewport_height` — càng gần biên Nam càng nguy hiểm |

### 2.2. Scalar Features — vector `(K,)` nối concat sau CNN

| # | Feature | Giá trị | Mô tả |
|:--:|:--------|:--------|:------|
| 0 | `robot_type` (one-hot) | 4 dims | [Factory, Scout, Worker, Miner] |
| 1 | `energy_ratio` | 0.0-1.0 | `energy / maxEnergy` của robot hiện tại |
| 2 | `move_cooldown` | 0.0-1.0 | `move_cd / movePeriod` |
| 3 | `build_cooldown` | 0.0-1.0 | `build_cd / 10` (chỉ Factory) |
| 4 | `jump_cooldown` | 0.0-1.0 | `jump_cd / 20` (chỉ Factory) |
| 5 | `distance_to_south` | 0.0-1.0 | `(row - southBound) / viewport_height` |
| 6 | `game_progress` | 0.0-1.0 | `current_step / 500` |
| 7 | `scroll_speed` | 0.0-1.0 | Tốc độ scroll hiện tại (normalized) |
| 8 | `total_team_energy` | 0.0-1.0 | Tổng energy phe mình / 5000 |
| 9 | `total_team_units` | 0.0-1.0 | Số unit phe mình / 20 |

**Tổng scalar dims: 4 (one-hot) + 9 = 13**

### 2.3. Xử lý Edge Cases

```python
def extract_local_view(obs, robot_col, robot_row, radius=5):
    """Trích xuất local view 11×11 quanh robot."""
    channels = np.zeros((12, 2*radius+1, 2*radius+1), dtype=np.float32)

    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            r, c = robot_row + dy, robot_col + dx
            ly, lx = dy + radius, dx + radius  # local coords

            # Ngoài map → tất cả kênh = 0 (mặc định)
            if c < 0 or c >= width or r < obs.southBound or r >= obs.northBound:
                continue

            idx = (r - obs.southBound) * width + c
            wall = obs.walls[idx]

            if wall == -1:  # Fog of war - chưa khám phá
                channels[4, ly, lx] = 0  # visibility = 0
                continue  # Mọi thông tin khác = 0

            # Đã khám phá
            channels[4, ly, lx] = 1
            channels[0, ly, lx] = 1 if (wall & 1) else 0  # North
            channels[1, ly, lx] = 1 if (wall & 2) else 0  # East
            channels[2, ly, lx] = 1 if (wall & 4) else 0  # South
            channels[3, ly, lx] = 1 if (wall & 8) else 0  # West

            # ... fill other channels from obs.robots, obs.crystals, etc.

    return channels
```

---

## 3. Thiết kế Chi Tiết Action Space

### 3.1. Unified Action Table (13 actions)

Tất cả robot types chia sẻ chung 1 action space kích thước **13**. Actions không hợp lệ cho type đó sẽ bị mask bằng 0.

| ID | Action | Factory | Scout | Worker | Miner |
|:--:|:-------|:-------:|:-----:|:------:|:-----:|
| 0 | `IDLE` | ✅ | ✅ | ✅ | ✅ |
| 1 | `NORTH` | ✅ | ✅ | ✅ | ✅ |
| 2 | `SOUTH` | ✅ | ✅ | ✅ | ✅ |
| 3 | `EAST` | ✅ | ✅ | ✅ | ✅ |
| 4 | `WEST` | ✅ | ✅ | ✅ | ✅ |
| 5 | `BUILD_SCOUT` | ✅ | ❌ | ❌ | ❌ |
| 6 | `BUILD_WORKER` | ✅ | ❌ | ❌ | ❌ |
| 7 | `BUILD_MINER` | ✅ | ❌ | ❌ | ❌ |
| 8 | `JUMP_NORTH` | ✅ | ❌ | ❌ | ❌ |
| 9 | `REMOVE_WALL` | ❌ | ❌ | ✅ | ❌ |
| 10 | `BUILD_WALL` | ❌ | ❌ | ✅ | ❌ |
| 11 | `TRANSFORM` | ❌ | ❌ | ❌ | ✅ |
| 12 | `TRANSFER` | ✅ | ✅ | ✅ | ✅ |

> [!NOTE]
> **Đơn giản hóa quan trọng**: Actions có hướng (REMOVE_NORTH/SOUTH/EAST/WEST, BUILD_WALL, TRANSFER, JUMP) được gộp thành 1 action ID. **Hướng được suy ra tự động bằng heuristic** (ví dụ: REMOVE_WALL → remove tường hướng Bắc nếu có, rồi thử E/W/S; TRANSFER → transfer đến robot gần nhất). Điều này giảm action space từ 25+ xuống 13 mà vẫn cover được mọi gameplay.

> [!IMPORTANT]
> **Alternative**: Nếu muốn agent tự chọn hướng chi tiết hơn, có thể dùng **Hierarchical Actions**: action ID chọn type (13), rồi sub-action chọn hướng (4). Nhưng điều này phức tạp hơn đáng kể.

### 3.2. Action Masking Logic

```python
def compute_action_mask(robot_data, obs, config):
    """Tạo mask [13] cho 1 robot. True = hợp lệ."""
    rtype, col, row, energy, _, move_cd, jump_cd, build_cd = robot_data
    mask = np.zeros(13, dtype=bool)

    idx = (row - obs.southBound) * config.width + col
    wall = obs.walls[idx] if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1 else 15

    # === IDLE luôn hợp lệ ===
    mask[0] = True

    # === Movement (actions 1-4) — nếu không có tường chắn và move_cd == 0 ===
    if move_cd == 0:
        if not (wall & 1): mask[1] = True  # NORTH
        if not (wall & 4): mask[2] = True  # SOUTH
        if not (wall & 2): mask[3] = True  # EAST
        if not (wall & 8): mask[4] = True  # WEST

    # === Factory-specific (actions 5-8) ===
    if rtype == 0:
        if build_cd == 0 and not (wall & 1):  # Spawn cell must be clear
            if energy >= config.scoutCost:  mask[5] = True   # BUILD_SCOUT
            if energy >= config.workerCost: mask[6] = True   # BUILD_WORKER
            if energy >= config.minerCost:  mask[7] = True   # BUILD_MINER
        if jump_cd == 0:
            # Check landing cell is on the board
            mask[8] = True  # JUMP_NORTH (add boundary check)

    # === Worker-specific (actions 9-10) ===
    if rtype == 2 and energy >= config.wallRemoveCost:
        if any_wall_adjacent(wall):     mask[9] = True   # REMOVE_WALL
        if any_open_adjacent(wall):     mask[10] = True  # BUILD_WALL

    # === Miner-specific (action 11) ===
    if rtype == 3:
        mining_key = f"{col},{row}"
        if mining_key in obs.miningNodes and energy >= config.transformCost:
            mask[11] = True  # TRANSFORM

    # === TRANSFER (action 12) — cần adjacent friendly robot ===
    if has_adjacent_friendly(col, row, obs):
        mask[12] = True

    return mask
```

---

## 4. Reward Shaping (Expanded)

### 4.1. Bảng Reward Đầy Đủ

| Category | Sự kiện | Reward | Rationale |
|:---------|:--------|:------:|:----------|
| **🏆 Game Result** | Thắng trận | `+10.0` | Terminal reward chính |
| | Thua trận | `-10.0` | Phạt thua |
| | Hòa | `0.0` | Neutral |
| **⚡ Energy** | Ăn Crystal | `+(value / 50)` | Khuyến khích thu thập (~0.2 - 1.0) |
| | Thu energy từ Mine mỗi turn | `+0.1` per mine owned | Mine = nguồn thu passive chính |
| | Xây Mine thành công (TRANSFORM) | `+3.0` | Đầu tư dài hạn, thưởng cao |
| **🔭 Exploration** | Khám phá ô mới (bất kỳ robot) | `+0.02` per ô | Khuyến khích mở fog nhưng không dominant |
| **⚔️ Combat** | Crush thắng robot đối phương | `+1.5` | Khuyến khích combat tấn công |
| | Mất robot bất kỳ (combat/scroll) | `-0.5` | Phạt nhẹ — đôi khi mất robot là trade hợp lý |
| | Friendly fire (robot mình đâm nhau) | `-3.0` | **Phạt nặng** — hoàn toàn có thể tránh |
| **🏃 Survival / Tempo** | Factory còn sống mỗi turn | `+0.01` | Baseline survival reward |
| | Robot quá gần southBound (< 3 ô) | `-0.1` per robot per turn | Cảnh báo scroll danger |
| | Factory bị scroll nuốt | `-5.0` | Catastrophic — gần như thua |
| **📈 Economy** | Delta energy (energy gain per turn) | `+0.001 × Δenergy` | Sparse signal cho growth |
| **🚫 Anti-stagnation** | Robot IDLE quá 5 turns liên tiếp | `-0.05` per turn | Tránh agent lười |

### 4.2. Reward Normalization

```python
# Clip total reward per step vào [-1, 1] để ổn định training
step_reward = np.clip(raw_reward, -1.0, 1.0)

# Exception: terminal rewards (win/loss) không bị clip
if done:
    step_reward = terminal_reward  # ±10.0
```

> [!TIP]
> **Curriculum reward tuning**: Ban đầu tăng trọng số exploration/energy, giảm dần khi agent đã biết basics. Dùng linear annealing cho các reward coefficients.

---

## 5. Neural Network Architecture

### 5.1. CNN-GRU Actor-Critic

```
Input: local_map (12, 11, 11)     Input: scalars (13,)
         │                              │
    ┌────▼────┐                         │
    │ Conv2d  │ 12→32, k=3, p=1        │
    │ BN+ReLU │                         │
    ├─────────┤                         │
    │ Conv2d  │ 32→64, k=3, p=1        │
    │ BN+ReLU │                         │
    ├─────────┤                         │
    │ Conv2d  │ 64→64, k=3, p=1        │
    │ BN+ReLU │                         │
    ├─────────┤                         │
    │ Flatten │ → 64×11×11 = 7744      │
    │ FC(256) │ → 256                   │
    └────┬────┘                         │
         │                              │
         └──────── concat ──────────────┘ → 256 + 13 = 269
                      │
                 ┌────▼────┐
                 │ FC(256) │
                 │  ReLU   │
                 ├─────────┤
                 │ GRU(256)│ ← hidden_state (cho partial observability)
                 └────┬────┘
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      ┌──────────┐       ┌──────────┐
      │ Actor    │       │ Critic   │
      │ FC(128)  │       │ FC(128)  │
      │ FC(13)   │       │ FC(1)    │
      │ +Masking │       │          │
      └──────────┘       └──────────┘
      π(a|s, mask)        V(s)
```

### 5.2. Action Masking trong Forward Pass

```python
def forward(self, local_map, scalars, action_mask, hidden_state=None):
    # CNN encode spatial features
    x = self.cnn(local_map)               # (B, 256)
    x = torch.cat([x, scalars], dim=-1)   # (B, 269)
    x = self.fc_shared(x)                 # (B, 256)

    # GRU for memory (POMDP)
    x, new_hidden = self.gru(x.unsqueeze(0), hidden_state)
    x = x.squeeze(0)                      # (B, 256)

    # Actor head with masking
    logits = self.actor(x)                # (B, 13)

    # Mask invalid actions: set logits to -inf
    logits = logits.masked_fill(~action_mask, float('-inf'))

    action_probs = F.softmax(logits, dim=-1)

    # Critic head
    value = self.critic(x)                # (B, 1)

    return action_probs, value, new_hidden
```

### 5.3. Tham số Model

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| CNN channels | 32→64→64 | Đủ capacity cho 11×11 input |
| GRU hidden | 256 | Balance giữa memory capacity và inference speed |
| Actor FC | 128→13 | Nhỏ gọn, 13 actions |
| Critic FC | 128→1 | Tương tự actor |
| Total params | **~2.5M** | Nhỏ gọn, inference nhanh trên CPU |

---

## 6. Training Pipeline

### 6.1. Giai đoạn 0: Curriculum Learning (Warm-up)

**Mục tiêu**: Agent học basics (tránh tường, tránh scroll, ăn crystal) trong môi trường đơn giản.

```python
# Curriculum levels
CURRICULUM = [
    {"scroll_speed": 0,    "fog": False, "enemy": None,      "desc": "No scroll, full vision"},
    {"scroll_speed": 0.25, "fog": False, "enemy": None,      "desc": "Slow scroll, no fog"},
    {"scroll_speed": 0.25, "fog": True,  "enemy": None,      "desc": "Slow scroll + fog"},
    {"scroll_speed": 0.25, "fog": True,  "enemy": "random",  "desc": "Slow scroll + random opponent"},
    {"scroll_speed": 1.0,  "fog": True,  "enemy": "random",  "desc": "Full speed + random"},
]
```

**Gate condition**: Chuyển level tiếp theo khi mean reward > threshold liên tiếp 100 episodes.

### 6.2. Giai đoạn 1: Huấn luyện vs Heuristic

```python
# Đối thủ = improved_agent.py (scout rush) + random
opponents = [
    "f:/Study/cloud/maze-crawler/improved_agent.py",  # Scout rush
    "f:/Study/cloud/maze-crawler/main.py",              # Worker rush
    "random",
]
# Chọn đối thủ ngẫu nhiên mỗi episode
```

**PPO Hyperparameters:**

| Parameter | Value | Note |
|:----------|:------|:-----|
| `learning_rate` | `3e-4` → `1e-5` (cosine decay) | |
| `n_steps` | 2048 | Rollout buffer size |
| `batch_size` | 256 | Mini-batch size |
| `n_epochs` | 4 | PPO epochs per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip |
| `ent_coef` | 0.01 → 0.001 (anneal) | Entropy bonus cho exploration |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `n_envs` | 8-16 | Parallel environments |
| `total_timesteps` | 10M - 50M | Tuỳ convergence |

### 6.3. Giai đoạn 2: Self-Play League

```python
class SelfPlayLeague:
    """
    Duy trì một pool gồm N phiên bản cũ của agent.
    Mỗi episode, chọn đối thủ ngẫu nhiên từ pool.
    Mỗi K steps, snapshot model hiện tại vào pool.
    """
    def __init__(self, pool_size=10, snapshot_interval=100_000):
        self.pool = []  # List of saved model paths
        self.pool_size = pool_size
        self.snapshot_interval = snapshot_interval

    def get_opponent(self):
        if not self.pool or random.random() < 0.2:
            return "random"  # 20% chance đấu random để tránh overfitting
        return random.choice(self.pool)

    def maybe_snapshot(self, model, step):
        if step % self.snapshot_interval == 0:
            path = f"checkpoints/model_step_{step}.zip"
            model.save(path)
            self.pool.append(path)
            if len(self.pool) > self.pool_size:
                self.pool.pop(0)  # Remove oldest
```

### 6.4. Monitoring & Logging

```python
# TensorBoard metrics to track
METRICS = [
    "rollout/ep_rew_mean",          # Mean episode reward
    "rollout/ep_len_mean",          # Mean episode length
    "custom/win_rate",              # % win vs opponents
    "custom/avg_energy",            # Average total energy
    "custom/avg_units",             # Average unit count
    "custom/mines_built",           # Mines per episode
    "custom/friendly_fire_count",   # Friendly fire incidents
    "custom/scroll_deaths",         # Deaths by scroll
    "train/policy_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
]
```

---

## 7. Kaggle Submission Pipeline

### 7.1. Model Export

```python
# Option A: PyTorch JIT (recommended)
traced = torch.jit.trace(model.policy, (dummy_map, dummy_scalar, dummy_mask, dummy_hidden))
traced.save("model_traced.pt")

# Option B: ONNX (faster CPU inference)
torch.onnx.export(
    model.policy,
    (dummy_map, dummy_scalar, dummy_mask, dummy_hidden),
    "model.onnx",
    opset_version=17,
    dynamic_axes={"local_map": {0: "batch"}, "scalars": {0: "batch"}}
)
```

### 7.2. Inference Agent (`main.py` cho submission)

```python
"""Submission agent — loads trained model, runs inference on CPU."""
import torch
import numpy as np

# Load model once at import time
MODEL = None
HIDDEN_STATES = {}  # {uid: hidden_state tensor}

def _load_model():
    global MODEL
    if MODEL is None:
        MODEL = torch.jit.load("model_traced.pt", map_location="cpu")
        MODEL.eval()

def agent(obs, config):
    _load_model()
    actions = {}

    my_robots = {uid: d for uid, d in obs.robots.items() if d[4] == obs.player}

    # Clean up hidden states for dead robots
    live_uids = set(my_robots.keys())
    for dead_uid in list(HIDDEN_STATES.keys()):
        if dead_uid not in live_uids:
            del HIDDEN_STATES[dead_uid]

    for uid, data in my_robots.items():
        # Extract observation
        local_map = extract_local_view(obs, data[1], data[2], config)
        scalars = extract_scalars(data, obs, config)
        mask = compute_action_mask(data, obs, config)
        hidden = HIDDEN_STATES.get(uid)

        # Inference
        with torch.no_grad():
            map_t = torch.from_numpy(local_map).unsqueeze(0)
            sc_t = torch.from_numpy(scalars).unsqueeze(0)
            mask_t = torch.from_numpy(mask).unsqueeze(0)

            probs, _, new_hidden = MODEL(map_t, sc_t, mask_t, hidden)
            action_id = probs.argmax(dim=-1).item()  # deterministic
            HIDDEN_STATES[uid] = new_hidden

        actions[uid] = ACTION_ID_TO_STRING[action_id]

    return actions
```

### 7.3. Submission Bundle

```bash
# File structure:
# main.py            — agent function (inference only)
# model_traced.pt    — traced PyTorch model (~10MB)
# obs_utils.py       — extract_local_view, compute_action_mask, etc.

tar -czf submission.tar.gz main.py model_traced.pt obs_utils.py
kaggle competitions submit maze-crawler -f submission.tar.gz -m "DRL PPO v1"
```

> [!WARNING]
> **Kaggle runtime**: PyTorch có sẵn trên Kaggle. Nhưng cần verify version compatibility. ONNX Runtime (`onnxruntime`) cũng có sẵn và nhanh hơn PyTorch inference ~2-3x trên CPU.

---

## 8. Project Structure

```
f:/Study/cloud/maze-crawler/
├── main.py                    # [MODIFY] Submission agent (inference)
├── improved_agent.py          # [KEEP] Heuristic opponent for training
├── test_agent.py              # [MODIFY] Update test script
│
├── drl/                       # [NEW] DRL training codebase
│   ├── __init__.py
│   ├── env_wrapper.py         # Gymnasium wrapper cho Crawl
│   ├── obs_utils.py           # extract_local_view, extract_scalars
│   ├── action_utils.py        # compute_action_mask, ACTION_ID_TO_STRING
│   ├── reward.py              # Reward shaping logic
│   ├── model.py               # CNN-GRU Actor-Critic network
│   ├── train.py               # Training loop (PPO + curriculum + self-play)
│   ├── self_play.py           # Self-play league manager
│   ├── config.py              # Hyperparameters & training config
│   └── evaluate.py            # Evaluation against various opponents
│
├── checkpoints/               # [NEW] Saved model checkpoints
├── logs/                      # [NEW] TensorBoard logs
└── submission/                # [NEW] Submission artifacts
    ├── main.py                # Copy of inference agent
    ├── obs_utils.py
    ├── action_utils.py
    └── model_traced.pt
```

---

## 9. Proposed Implementation Order

### Phase 1: Foundation (Week 1-2)

| Task | File | Description |
|:-----|:-----|:------------|
| 1.1 | `drl/obs_utils.py` | `extract_local_view()` + `extract_scalars()` + unit tests |
| 1.2 | `drl/action_utils.py` | `compute_action_mask()` + `ACTION_ID_TO_STRING` + unit tests |
| 1.3 | `drl/reward.py` | Reward shaping function + configurable coefficients |
| 1.4 | `drl/env_wrapper.py` | `CrawlGymnasiumEnv(gymnasium.Env)` — single-agent wrapper treating each robot as independent agent |
| 1.5 | | **Smoke test**: wrapper runs with random actions for 100 episodes without crashing |

### Phase 2: Model & Training (Week 3-4)

| Task | File | Description |
|:-----|:-----|:------------|
| 2.1 | `drl/model.py` | CNN-GRU Actor-Critic với action masking |
| 2.2 | `drl/config.py` | Hyperparameter configs |
| 2.3 | `drl/train.py` | MaskablePPO training loop + TensorBoard logging |
| 2.4 | | **Milestone**: Agent beats `random` opponent >80% win rate |

### Phase 3: Heuristic Training (Week 4-5)

| Task | File | Description |
|:-----|:-----|:------------|
| 3.1 | `drl/train.py` | Add curriculum learning |
| 3.2 | `drl/train.py` | Train vs `improved_agent.py` + `main.py` |
| 3.3 | `drl/evaluate.py` | Evaluation suite (win rate, avg energy, avg units) |
| 3.4 | | **Milestone**: Agent beats heuristic opponents >60% |

### Phase 4: Self-Play (Week 5-7)

| Task | File | Description |
|:-----|:-----|:------------|
| 4.1 | `drl/self_play.py` | Self-play league manager |
| 4.2 | `drl/train.py` | Integrate self-play into training loop |
| 4.3 | | **Milestone**: Agent shows emergent strategies (mining, combat avoidance) |

### Phase 5: Submission & Polish (Week 7-8)

| Task | File | Description |
|:-----|:-----|:------------|
| 5.1 | `main.py` | Inference-only agent with model loading |
| 5.2 | | Export model → JIT/ONNX |
| 5.3 | | Bundle + submit to Kaggle |
| 5.4 | | Iterate based on leaderboard performance |

---

## 10. Verification Plan

### Automated Tests

```bash
# Unit tests for observation/action utilities
python -m pytest drl/tests/ -v

# Smoke test: wrapper runs without crash
python -c "
from drl.env_wrapper import CrawlGymnasiumEnv
env = CrawlGymnasiumEnv(opponent='random')
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done: obs, info = env.reset()
print('Smoke test passed!')
"

# Training convergence test (short run)
python drl/train.py --total_timesteps 100000 --opponent random

# Win rate evaluation
python drl/evaluate.py --model checkpoints/latest.zip --opponent improved_agent.py --n_games 100
```

### Manual Verification

- Xem TensorBoard logs: `tensorboard --logdir logs/`
- Replay games trên Kaggle sau khi submit
- So sánh win rate trên leaderboard

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Bạn muốn bắt đầu với **MaskablePPO (sb3-contrib)** hay viết PPO custom từ CleanRL? MaskablePPO dễ setup hơn nhưng ít flexible. CleanRL cho full control nhưng nhiều code hơn.

> [!IMPORTANT]
> **Q2**: Có muốn thêm **GRU** (memory) ngay từ đầu hay thử **MLP-only** trước rồi add GRU sau? GRU giúp handle fog-of-war tốt hơn nhưng training chậm hơn ~30%.

> [!NOTE]
> **Q3**: Reward coefficients trong bảng ở Section 4 là giá trị khởi đầu. Có muốn implement auto-tuning (adaptive reward coefficients) hay manual tuning?
