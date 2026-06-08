# Maze Crawler DRL Training Suite

Complete implementation of Deep Reinforcement Learning (PPO) agent for Kaggle's Crawl competition.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-drl.txt
```

### 2. Run Training

```bash
# Basic training vs random opponent
python -m drl.train --total-timesteps 10000000 --opponent random --num-envs 8

# With GPU
python -m drl.train --total-timesteps 10000000 --device cuda

# Load from checkpoint
python -m drl.train --checkpoint ./checkpoints/model_step_1000000.pt
```

### 3. Evaluate Model

```bash
python -m drl.evaluate \
  --model ./checkpoints/model_step_1000000.pt \
  --opponent random \
  --num-episodes 50 \
  --device cuda
```

### 4. Export for Submission

```bash
python -c "
import torch
from drl.model import CrawlActorCritic

model = CrawlActorCritic()
checkpoint = torch.load('checkpoints/model_step_1000000.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# JIT trace
dummy_map = torch.randn(1, 12, 11, 11)
dummy_scalar = torch.randn(1, 13)
dummy_mask = torch.ones(1, 13, dtype=torch.bool)
dummy_hidden = torch.randn(1, 256)

traced = torch.jit.trace(
    model,
    (dummy_map, dummy_scalar, dummy_mask, dummy_hidden)
)
traced.save('submission/model_traced.pt')
print('Model exported!')
"
```

## Architecture

### Observation Space (12 channels, 11×11 local view)

| Channel | Description                  |
| ------- | ---------------------------- |
| 0-3     | Wall directions (N/E/S/W)    |
| 4       | Visibility (fog-of-war mask) |
| 5       | My robots (type 0-4)         |
| 6       | My robot energy (0.0-1.0)    |
| 7       | Enemy robots (type)          |
| 8       | Crystals (energy normalized) |
| 9       | Mining nodes                 |
| 10      | Mines (-1/0/+1)              |
| 11      | Scroll danger                |

**+ 13 scalar features:** robot type, energy, cooldowns, distance to scroll, game progress, etc.

### Action Space (13 unified actions)

All robot types share the same action space with type-specific masking:

| ID  | Action                   | Valid For |
| --- | ------------------------ | --------- |
| 0   | IDLE                     | All       |
| 1-4 | NORTH/SOUTH/EAST/WEST    | All       |
| 5-7 | BUILD_SCOUT/WORKER/MINER | Factory   |
| 8   | JUMP_NORTH               | Factory   |
| 9   | REMOVE_WALL              | Worker    |
| 10  | BUILD_WALL               | Worker    |
| 11  | TRANSFORM                | Miner     |
| 12  | TRANSFER                 | All       |

### Network Architecture

```
Input: Local Map (12, 11, 11) + Scalars (13)
  ↓
CNN: 12→32→64→64 (3 Conv2D layers, BN, ReLU)
  ↓
Flatten + Concat Scalars → 256
  ↓
FC: 256 → 256 (ReLU)
  ↓
GRU: 256 cells (for partial observability/memory)
  ↓
┌─────────────┬──────────────┐
Actor Head:   Critic Head:
128 → 13      128 → 1
```

**Total params:** ~2.5M (lightweight for CPU inference)

## Training Configuration

### Key Hyperparameters

- **PPO Clip:** 0.2
- **Learning Rate:** 3e-4 → 1e-5 (cosine decay)
- **Gamma:** 0.99
- **GAE Lambda:** 0.95
- **Batch Size:** 256
- **Rollout Steps:** 2048
- **Entropy Coef:** 0.01 → 0.001 (annealing)

### Reward Shaping

| Event             | Reward                 |
| ----------------- | ---------------------- |
| Win               | +10.0                  |
| Loss              | -10.0                  |
| Crystal collected | +(value/50)            |
| Mine income       | +0.1 per mine per turn |
| Mine built        | +3.0                   |
| Crush victory     | +1.5                   |
| Robot loss        | -0.5                   |
| Friendly fire     | -3.0                   |
| Survival          | +0.01 per turn         |

## Project Structure

```
drl/
├── __init__.py
├── obs_utils.py          # Observation extraction
├── action_utils.py       # Action masking & resolution
├── reward.py             # Reward shaping
├── env_wrapper.py        # Gymnasium wrapper
├── model.py              # CNN-GRU Actor-Critic
├── config.py             # Hyperparameters
├── train.py              # Training loop (PPO)
├── evaluate.py           # Evaluation utilities
├── self_play.py          # Self-play league
└── tests/
    ├── test_obs.py
    ├── test_action.py
    └── test_env.py

checkpoints/             # Saved model checkpoints
logs/                    # TensorBoard logs
submission/              # Submission artifacts
```

## Running Tests

```bash
# Test obs extraction
python -m pytest drl/tests/test_obs.py -v

# Test action masking
python -m pytest drl/tests/test_action.py -v

# Smoke test environment
python -c "
from drl.env_wrapper import CrawlGymnasiumEnv
env = CrawlGymnasiumEnv(opponent='random')
obs, _ = env.reset()
for _ in range(100):
    actions = {uid: 0 for uid in obs}  # IDLE
    obs, reward, done, truncated, info = env.step(actions)
    if done or truncated:
        break
print('Smoke test passed!')
"
```

## Monitoring Training

```bash
# View TensorBoard
tensorboard --logdir logs/

# Then open http://localhost:6006 in browser
```

## Expected Convergence

- **Phase 1 (Random opponent):** 1-2M timesteps → >80% win rate
- **Phase 2 (Heuristic opponents):** 5-10M timesteps → >60% win rate
- **Phase 3 (Self-play):** 20-50M timesteps → Emergent strategies

## Troubleshooting

### CUDA Out of Memory
- Reduce `num_envs` from 8 to 4 or 2
- Reduce `batch_size` from 256 to 128

### Slow Training
- Increase `num_envs` (if VRAM permits)
- Use `--device cuda` instead of CPU

### Model Not Improving
- Increase entropy coefficient for more exploration
- Reduce learning rate for stability
- Check reward shaping coefficients

## Next Steps (Phase 2+)

1. **Curriculum Learning:** Gradually increase difficulty (scroll speed, fog-of-war)
2. **Self-Play:** Train against pool of older models for emergent strategies
3. **Fine-tuning:** Train longer vs specific heuristic opponents
4. **Submission:** Export to JIT and bundle with inference code

## References

- Original paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- CleanRL implementation: https://github.com/vwxyzjn/cleanrl
- Kaggle Crawl: https://www.kaggle.com/competitions/maze-crawler
