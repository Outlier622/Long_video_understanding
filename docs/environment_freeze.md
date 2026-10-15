# Frozen serving environment — 2026-10-14

“冻结运行环境”是保存一份可复现的运行基线：使用哪些软件、哪份代码、
哪个模型、哪些启动参数。以后可以继续开发和升级，但应该保留这份基线，
为新版本另外建立快照。它不意味着当前模型的速度、质量或可靠性已经达标。

本次保持常驻服务运行，未安装或升级任何运行依赖，也没有改变模型文件。

## 已固定的内容

- 完整运行镜像：`sha256:ad8633e18d7b966abfbb87adefe459147704cf3ba0dfeef18fb05f21f24e6368`。
- 固定基础镜像：`sha256:421bcd51d2db69a916c2bd375367bb51b4f7e8c28200ee4091414f8d539de2df`。
  `Dockerfile.serving` 已改为按 digest 引用它，构建验证通过。
- 231 个 Python 分发包的精确版本，以及完整 OS 包清单。
- 37 个仓库 Python 文件的内容哈希，以及基础镜像继承的 4 个旧 Worker 文件哈希。
  这些继承文件实际存在于镜像中，包含在基础镜像和归档中。
- 推理/恢复环境变量、单进程启动命令、GPU 资源申请、端口、共享内存和重启策略。
- 13 个模型文件的完整 SHA256，包括配置、tokenizer、权重索引及两个 BF16 权重分片。
  完整权重内容已经核验，不只核验 safetensors 头部。

快照目录为 `environment/serving-20261014/`：

| 文件 | 用途 |
|---|---|
| `manifest.json` | 镜像身份、归档哈希、文件校验信息和主机基线 |
| `runtime.lock.json` | Python 版本、包版本、源码哈希和继承文件 |
| `requirements.constraints.txt` | 全部 Python 版本的核对约束 |
| `pip-freeze.raw.txt` | 原始导出记录，保留 Conda 构建来源信息 |
| `os-packages.lock.txt` | 系统包版本记录 |
| `model.lock.json` | 模型文件大小与完整内容哈希 |
| `compose.frozen.json` | 不重新构建、不拉取浮动标签的冻结启动配置 |

`pip freeze` 里有 Conda 构建机的 `file:///home/conda/...` 临时路径，直接
`pip install -r` 无法在新机器重建。约束文件也包含由基础镜像供应的 Conda
和 CUDA 包，**不能把它当作从空白 Python 环境安装的完整配方**。本次精确
复现依靠冻结的镜像，包清单用于核对和解释版本差异。

当前 Git 开启了 Windows 换行转换。`.gitattributes` 保留冻结清单的原始字节，
以维持文件 SHA256；仓库源码/构建文件另存 LF 规范化指纹，允许纯 CRLF/LF
差异。容器内源码仍使用原始字节哈希，原始镜像也按完整 digest 固定。

## 本地镜像备份

`evaluation/runs/environment_freeze_20261014/serving-images.tar` 包含运行镜像
和固定基础镜像，共享层只保存一次。大小 4,325,050,880 字节（约 4.3 GB）。

SHA256：

```text
1ffba0e752f37a3d711dfbb97697c296a1b306f0f6c1728b9fa6391a7aa20ec4
```

归档位于 Git 忽略的运行目录，不应提交到 GitHub。源码与版本清单可以进入
版本控制。复制项目或换机器时，需要另外带走镜像归档、模型和视频数据。

归档不包含挂载在镜像外面的模型/视频，也不包含 Windows、WSL 内核、
Docker Desktop 或 GPU 驱动。当前记录了 Docker Engine 28.0.4，以及 GPU/
驱动基线；换 GPU 或驱动后要重新验证运行兼容性和性能。相同镜像不保证
不同硬件得到相同延迟。

## 核验现有环境

在仓库根目录执行，主机 Python 只需标准库；不要求主机安装 vLLM：

```powershell
python scripts/verify_serving_environment.py `
  --archive evaluation/runs/environment_freeze_20261014/serving-images.tar `
  --container videothinker-serving
```

工具检查快照完整性、镜像身份、源码、13 个模型文件的完整哈希，以及运行
容器中的包版本、OS 包、环境变量和资源/启动配置。带 `--archive` 时也检查
整个归档。核验模型与归档会顺序读取较大文件，需要一定磁盘 IO。

这是版本/内容核验，不是模型质量、并发或故障恢复测试。SHA256 可以发现
与受信任基线的差异；快照没有签名，不构成发布者身份认证。

## 在另一台机器还原

先准备支持 Linux/amd64 容器的 Docker GPU 环境，把仓库、归档、模型、数据放到对应位置。
通过环境变量指定外部目录，保持冻结配置文件本身不变：

```powershell
$env:VIDEO_MODEL_DIR = 'D:/projects/VideoThinker-R1-3B'
$env:VIDEO_DATA_DIR = 'D:/projects/longvideo'
docker image load -i evaluation/runs/environment_freeze_20261014/serving-images.tar
python scripts/verify_serving_environment.py --archive evaluation/runs/environment_freeze_20261014/serving-images.tar
docker compose -f environment/serving-20261014/compose.frozen.json up -d --no-build
```

如果普通配置启动的服务已经在运行，冻结配置沿用同一 Compose 项目和
容器名称；切换配置可能重新创建服务并重新加载模型。启动后检查
`http://127.0.0.1:8000/health/ready`，再用 `--container videothinker-serving`
核验实际运行实例，并运行 HTTP 功能测试。

停止冻结配置：

```powershell
docker compose -f environment/serving-20261014/compose.frozen.json down
```

开发构建仍可使用 `compose.serving.yml`。修改源码、依赖、模型或参数后，
旧快照核验失败是预期行为；为新基线创建新的快照目录，保留旧基线。
导出工具 `scripts/snapshot_serving_environment.py` 支持 `--baseline-date` 和
`--image-tag`；同一天建立多个版本时指定不同标签，工具会拒绝覆盖另一个
镜像的发布标签。模型/数据目录从当前容器的已验证挂载中读取。视频内容
本次没有冻结，后续性能测试还需要另外固定输入数据和采样方式。
`scripts/Dockerfile.vllm-validation` 和 `Dockerfile.worker` 是此前的开发/
bootstrap 配方，仍含部分浮动依赖，不能替代本次归档来精确还原基础环境。

## 验证证据

- 正在运行的服务与快照一致：Python/OS 包、源码、启动配置、完整模型
  哈希、归档哈希全部通过，GPU/驱动基线也匹配。
- 将临时副本的 `packages.vllm` 改为 `0.0.0` 后，核验工具正确拒绝副本。
  原始快照没有被修改。
- 固定基础 digest 的开发构建完成，运行层复用；镜像外层 provenance 会因
  构建时间变化，因此保存归档的原始镜像 ID 用于精确身份还原。
- 冻结 Compose 配置解析通过；常驻服务保持运行，没有因冻结而重启。
- 镜像归档已重新加载；运行/基础镜像的不可变身份保持一致，冻结镜像的
  Python 模块在断网容器中导入成功。

机器可读记录在 `evaluation/runs/environment_freeze_20261014/verification.json`
和 `tamper_check.json`；最终跨平台指纹核验为 `final_portable_verification.json`。
镜像归档重新加载的结果在同目录 `restore_verification.json`。
