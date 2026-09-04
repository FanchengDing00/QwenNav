# 在 QwenNav/conda_envs 中安装三个 Conda 环境

本教程为主动观察 Gate 实验准备三个彼此隔离的环境：

| 目录 | Python | 用途 |
|---|---:|---|
| `QwenNav/conda_envs/qwennav_model` | 3.11 | 原始 LightNav 模型与评估客户端 |
| `QwenNav/conda_envs/qwennav_habitat` | 3.9 | Habitat-Sim、Habitat-Lab 与环境服务 |
| `QwenNav/conda_envs/qwen3vl_py310` | 3.10 | 冻结的 Qwen3-VL-4B-Instruct Gate |

这些环境使用 Conda 的路径环境（prefix environment），不会安装到
`~/.conda/envs/`。仓库已忽略根目录下的 `conda_envs/`，环境文件不会被 Git
提交。模型参数、数据集和评估输出仍需单独准备，不包含在环境中。

## 1. 固定仓库和环境目录

先进入服务器上的 **QwenNav 仓库根目录**。以下命令以当前仓库的真实绝对路径
构造环境目录，因此即使仓库位置与本机不同，也会始终安装到
`QwenNav/conda_envs/`：

```bash
cd /path/to/QwenNav
QWENNAV_REPO_ROOT="$(pwd -P)"
QWENNAV_ENV_ROOT="$QWENNAV_REPO_ROOT/conda_envs"
mkdir -p "$QWENNAV_ENV_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
```

可检查目标路径：

```bash
printf '%s\n' "$QWENNAV_ENV_ROOT"
```

输出必须以 `/QwenNav/conda_envs` 结尾。后续步骤应在同一个终端中执行，以保留
`QWENNAV_REPO_ROOT` 和 `QWENNAV_ENV_ROOT` 两个变量。

## 2. 安装 LightNav 模型环境

```bash
conda create --prefix "$QWENNAV_ENV_ROOT/qwennav_model" python=3.11 pip -y
conda activate "$QWENNAV_ENV_ROOT/qwennav_model"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$QWENNAV_REPO_ROOT[vllm,video,habitat]"
```

这一步安装仓库自身以及正式评估使用的 vLLM、视频和 Habitat 客户端依赖。
`pyproject.toml` 会固定关键版本，例如 `transformers==5.8.0`、
`vllm==0.19.1` 和 `nvidia-cutlass-dsl==4.5.2`。不要在此环境中安装 Gate 的旧版
Transformers 或 PyTorch。

验证：

```bash
python -c "import torch, transformers, vllm, lightnav; print('torch', torch.__version__); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('cuda', torch.cuda.is_available())"
conda deactivate
```

## 3. 安装 Habitat 环境

```bash
conda env create \
  --prefix "$QWENNAV_ENV_ROOT/qwennav_habitat" \
  --file "$QWENNAV_REPO_ROOT/habitat_server/environment.yml"
conda activate "$QWENNAV_ENV_ROOT/qwennav_habitat"

python -m pip install --no-deps "habitat-lab==0.3.20231024"
python -m pip install --force-reinstall "numpy>=1.20,<1.24"
python -m pip install -e "$QWENNAV_REPO_ROOT/habitat_server"
```

`habitat_server/environment.yml` 安装官方评估所需的 headless Habitat-Sim 0.3.1。
Habitat-Lab 使用 `--no-deps` 是为了防止 pip 将 NumPy 自动升级到 2.x；最后再次
固定 NumPy 1.23.x 是有意为之。如果 pip 提示新版 `dtw-python` 偏好 NumPy 2.x，
本项目仍应保留 `<1.24`，导航指标所需的 `fastdtw` 已单独安装。

验证：

```bash
python -c "import numpy, habitat, habitat_sim; import lightnav_habitat.serve; print('numpy', numpy.__version__); print('habitat imports: OK')"
conda deactivate
```

## 4. 安装 Qwen3-VL Gate 环境

```bash
conda create --prefix "$QWENNAV_ENV_ROOT/qwen3vl_py310" python=3.10 pip -y
conda activate "$QWENNAV_ENV_ROOT/qwen3vl_py310"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --requirement "$QWENNAV_REPO_ROOT/experiments/active_observation_gate/requirements-qwen-gate.txt"
```

Gate 使用与 LightNav 完全独立的 PyTorch/Transformers 组合。依赖清单固定为本实验
已经验证过的 CUDA 12.4 组合。Gate 通过 Python 标准库 socket 与评估进程通信，
因此这个环境不需要 Habitat、LightNav 或 `pyzmq`。

### 安装 FlashAttention 2

RTX 4090 服务器需要可用的 CUDA 12.4 Toolkit（不仅是 NVIDIA 驱动）。如 Toolkit
位于常见路径，可执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"

nvcc --version
python -c "import torch; print('torch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
MAX_JOBS=4 python -m pip install --no-build-isolation "flash-attn==2.7.4.post1"
```

`nvcc` 与 `torch.version.cuda` 都应显示 12.4。若服务器的 CUDA Toolkit 安装在其他
位置，只调整 `CUDA_HOME`；不要改动系统 CUDA 文件。`MAX_JOBS=4` 用于限制编译时
的内存和 CPU 峰值，可依据服务器资源调整。

验证完整 Gate 环境：

```bash
python -c "import torch, transformers, flash_attn, cv2, numpy; print('torch', torch.__version__); print('transformers', transformers.__version__); print('flash_attn', flash_attn.__version__); print('opencv', cv2.__version__); print('numpy', numpy.__version__); print('cuda', torch.cuda.is_available())"
conda deactivate
```

## 5. 确认三个路径环境

```bash
conda env list
du -sh "$QWENNAV_ENV_ROOT"/*
```

列表中应出现以下三个绝对路径：

```text
.../QwenNav/conda_envs/qwennav_model
.../QwenNav/conda_envs/qwennav_habitat
.../QwenNav/conda_envs/qwen3vl_py310
```

路径环境应通过绝对路径激活，例如：

```bash
conda activate "$QWENNAV_ENV_ROOT/qwennav_model"
```

不要使用 `conda activate qwennav_model`；后者查找的是 Conda 默认环境目录中的命名
环境，不是仓库内的 prefix 环境。

## 6. 准备数据目录并建立软连接

评估配置固定从仓库根目录下的 `data/` 读取数据，但不需要把大型数据集复制进 Git
仓库。推荐把 R2R、RxR 和 MP3D 集中放在服务器的数据盘中，然后只为整个
`QwenNav/data` 建立 **一个软连接**。

外部数据目录必须整理为：

```text
QwenNav_Dataset/
├── datasets/
│   ├── R2R_VLNCE_v1-3_preprocessed/
│   │   └── val_unseen/
│   │       ├── val_unseen.json.gz
│   │       └── val_unseen_gt.json.gz
│   └── RxR_VLNCE_v0/
│       └── val_unseen/
│           ├── val_unseen_guide.json.gz
│           └── val_unseen_guide_gt.json.gz
└── scene_datasets/
    └── mp3d/
        ├── 17DRP5sb8fy/
        │   ├── 17DRP5sb8fy.glb
        │   ├── 17DRP5sb8fy.navmesh
        │   └── ...
        └── ...
```

假设服务器数据实际存放在 `/path/to/datasets/QwenNav_Dataset`，从仓库根目录执行：

```bash
cd /path/to/QwenNav
QWENNAV_REPO_ROOT="$(pwd -P)"
QWENNAV_DATA_ROOT=/path/to/datasets/QwenNav_Dataset

test -d "$QWENNAV_DATA_ROOT/datasets" \
  || { echo "缺少数据目录: $QWENNAV_DATA_ROOT/datasets"; exit 1; }
test -d "$QWENNAV_DATA_ROOT/scene_datasets/mp3d" \
  || { echo "缺少 MP3D: $QWENNAV_DATA_ROOT/scene_datasets/mp3d"; exit 1; }
test ! -e "$QWENNAV_REPO_ROOT/data" && test ! -L "$QWENNAV_REPO_ROOT/data" \
  || { echo "QwenNav/data 已存在，请先人工确认，不要覆盖"; exit 1; }

ln -s "$QWENNAV_DATA_ROOT" "$QWENNAV_REPO_ROOT/data"
```

这里必须将 `QWENNAV_DATA_ROOT` 换成服务器上的真实绝对路径。命令会在目标已存在
时主动退出，不会覆盖已有的真实目录或软连接。不要在路径末尾写成
`QwenNav_Dataset/` 后再附加通配符，否则可能错误地为每个子目录分别建链接。

确认软连接指向及评估必需文件：

```bash
ls -ld "$QWENNAV_REPO_ROOT/data"
readlink -f "$QWENNAV_REPO_ROOT/data"

test -f "$QWENNAV_REPO_ROOT/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
test -f "$QWENNAV_REPO_ROOT/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz"
test -f "$QWENNAV_REPO_ROOT/data/datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz"
test -f "$QWENNAV_REPO_ROOT/data/datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json.gz"
find "$QWENNAV_REPO_ROOT/data/scene_datasets/mp3d" -mindepth 2 -maxdepth 2 -name '*.glb' | head
```

四条 `test -f` 命令均应静默成功，最后一条应打印若干 MP3D `.glb` 文件。R2R 和
RxR 共用同一套 MP3D scene dataset，不需要重复存储。仓库根目录的 `/data/` 已在
`.gitignore` 中，因此软连接本身及其指向的大型数据都不会被 Git 提交。

如果服务器已经有完整的 `QwenNav_Dataset`，只需整理成上述结构并建立链接；不要
再次下载或复制。若 `QwenNav/data` 已经是正确链接，则保留它，不要重复执行
`ln -s`。

## 7. 让主动观察实验使用这三个环境

每次登录服务器后，从 QwenNav 根目录执行：

```bash
QWENNAV_REPO_ROOT="$(pwd -P)"
QWENNAV_ENV_ROOT="$QWENNAV_REPO_ROOT/conda_envs"

export LIGHTNAV_PYTHON="$QWENNAV_ENV_ROOT/qwennav_model/bin/python"
export GATE_PYTHON="$QWENNAV_ENV_ROOT/qwen3vl_py310/bin/python"
export HABITAT_ENV="$QWENNAV_ENV_ROOT/qwennav_habitat"
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
```

然后设置可见 GPU 和 Hugging Face 缓存，再启动实验：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/path/to/your/huggingface/cache

./experiments/active_observation_gate/eval_r2r.sh \
  checkpoints/LightNav-0 \
  gate_h2_4gpu
```

脚本会为每张可见 GPU 启动一个 Habitat、LightNav 和 Qwen Gate shard。这里设置的
四个环境变量只负责告诉现有实验启动器使用哪个 Python/Conda 环境，不会改变
LightNav 的模型参数、Prompt、SlowFast 历史、视频预算或正式评估配置。

## 8. 常见问题

- **磁盘空间**：三个环境合计可能占用数十 GB，安装前先检查 `df -h`。
- **模型与数据不随 Git 同步**：服务器仍需单独准备 `checkpoints/`、`data/` 和
  Hugging Face 缓存。
- **FlashAttention 编译失败**：优先检查 `nvcc --version`、`CUDA_HOME`、
  `torch.version.cuda` 是否一致，以及系统是否安装 C++ 编译器。
- **环境已存在**：`conda create --prefix`/`conda env create --prefix` 会拒绝覆盖，
  不要直接重建已有目录；先验证已有环境，确实需要删除时再明确处理对应的单个
  prefix。
