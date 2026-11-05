# SuperMimic-Agent-WFArm

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/Simulation-MuJoCo-009688)](https://mujoco.org/)
[![Status](https://img.shields.io/badge/Status-Exploratory-orange)](docs/PROJECT_ANALYSIS_CN.md)

**[English](README.md) · 简体中文**

**这是一个面向四轮足机械臂控制的探索类开源项目，欢迎大家交流与共同探索。**

项目围绕一个问题展开：轮、腿和机械臂共享同一套机器人动力学时，能否先学到一个可微的世界模型，再利用这个模型训练末端控制策略？代码覆盖轮腿臂协同控制、固定底盘六轴机械臂，以及参考动作生成与物理筛选。G1 动作跟踪代码保留为技术来源，四轮足机械臂是主要研究对象。

学习机制来自 [SuperMimic](https://arxiv.org/abs/2610.28480)：从真实仿真转移学习动力学，通过短时域想象展开和时间反向传播优化策略。[MimicAgent](https://arxiv.org/abs/2609.24145) 提供了参考生成的思路：将任务和阶段计划转成轨迹程序，再根据检查结果修正。

本仓库发布训练、评估、参考生成代码和使用说明。机器人 URDF、网格、标定、训练数据、模型权重、视频及内部实验文档不随仓库分发。没有私有机器人资源也可以运行算法单元测试；完整机器人仿真需要另外提供有权使用的资产。

---

**图片**：<img width="1440" height="540" alt="fixed_arm_showcase_contact_sheet" src="https://github.com/user-attachments/assets/fec88e0f-fa53-402b-a322-af3309023700" />


**视频**：

---


## 项目名称

- **SuperMimic**：学习可微动力学，并通过 BPTT 优化控制策略。
- **Agent**：保留任务、阶段计划、轨迹代码和诊断反馈的参考生成接口。目前由 coding agent 通过文件提交程序，尚未接入持续运行的外部 LLM 服务。
- **WFArm**：四轮足平台及其机械臂，同时研究整机协同和固定底盘机械臂任务。

## 项目包含什么

| 模块 | 任务和接口 | 主要代码 |
|---|---|---|
| 轮腿臂全身控制 | 一个 Actor 联合输出 12 路腿部位置、4 路轮速和 6 路机械臂位置命令 | `src/supermimic/wf/` |
| 固定底盘机械臂基线 | TCP 位姿到达与轨迹跟踪，输出六个限速关节位置目标 | `frameworks/fixed_base_arm/arm_smc/` |
| 结构化机械臂控制 | 碰撞检查的到达参考、学习的任务反馈增益、保留的姿态策略 | `frameworks/fixed_base_arm/variants/supermimic_robust_control_v7/arm_smc/` |
| 参考生成前端 | 任务描述与计划、轨迹程序、运动学检查、反馈修正、真实物理筛选 | `src/supermimic/mimicagent/` |
| 训练与评估 | 模型拟合、BC 辅助、BPTT、Critic、独立目标评估和失败诊断 | 各模块的训练和评估入口 |

全身策略与固定底盘策略是两种不同设置。前者研究自由底盘上的轮腿臂协同，后者将底盘和腿部真正固定，只保留六个机械臂自由度。固定底盘的结构化分支同时使用参考规划和解析雅可比反馈；评估和介绍时应保留这一前提。

本项目仍处于探索阶段。仓库提供算法和工程实现，不将局部仿真表现扩展成全工作空间或实机能力的结论。

## MimicAgent × SuperMimic：参考生成与策略学习的联动

两者通过参考数据接口连接：

1. **MimicAgent 风格前端**根据任务和阶段计划生成轨迹程序，执行运动学检查，并将诊断反馈交给下一版程序。
2. **真实物理执行与筛选**将提案转换为电机命令。独立检查通过后，实际状态和动作序列才成为示范数据。
3. **SuperMimic 风格后端**把这些参考用于模型拟合、可选 BC 和想象状态跟踪辅助，同时继续采集当前学生的真实转移，通过 BPTT 更新策略。
4. **独立物理评估**检查学生是否完成任务，与参考生成器、离线教师的成绩分别统计。

前端提供参考提案，世界模型与 BPTT 负责策略学习。这是本项目已经接通的技术关系；原论文完整基准和参考生成对最终学生的收益仍需要另行验证。

## 算法原理

### 1. 用真实仿真数据学习世界模型

真实物理引擎负责产生状态转移，神经网络负责拟合可微近似：

$$
\hat{s}_{t+1}=F_\theta(\hat{s}_t,a_t,\xi).
$$

$s_t$ 是机器人状态，$a_t$ 是实际执行的命令，$\xi$ 表示摩擦等环境条件。网络预测有界残差，再与显式积分结合。训练采用多步自回归展开，损失包括状态误差和失败概率：

$$
\mathcal L_{\mathrm{WM}}=\frac1K\sum_{k=1}^{K}
\left[\frac1m\left\|E(\hat s_{t+k},s_{t+k})\right\|_2^2
+\beta\operatorname{BCE}(\hat d_{t+k},d_{t+k})\right].
$$

$E$ 对不同状态量做尺度归一化，$m$ 是误差向量维数；全身分支中 $\beta=0.1$。轮子的累计转角不进入全身模型的状态误差，轮速仍然保留。

### 2. 在模型里展开，在真实物理里检验

可微正运动学提供末端位姿、速度和任务奖励。更新 Actor 时冻结世界模型与 Critic 的参数，同时保留状态和动作到回报的梯度：

$$
J(\phi)=\mathbb E\left[
\sum_{t=0}^{H-1}\gamma^t C_t r_{t+1}
+\gamma^H C_H V_\psi(\hat s_H,g_H,\xi)
\right],\qquad
C_t=\prod_{j=1}^{t}(1-\hat d_j).
$$

Actor 通过 BPTT 提高短时域回报，并可加入 BC 和参考状态跟踪辅助。Critic 用停止梯度的 $\lambda$-return 训练。较短展开减少模型误差积累，终端价值补充时域之外的回报估计。

### 3. 六轴机械臂采用限速位置接口

固定底盘机械臂内部可以输出归一化目标变化率，真正交给下游的是六个关节位置目标：

$$
q^{\mathrm{cmd}}_{t+1}
=\operatorname{clip}_{\mathcal Q}
\left(q^{\mathrm{cmd}}_t+\Delta t\,v_{\max}\,\rho_t\right),
\qquad \rho_t\in[-1,1]^6.
$$

结构化分支用有界正增益调整任务空间反馈，经阻尼雅可比映射成关节运动，并保留姿态或规划参考项。具体公式、坐标约定和代码对应关系见[中文项目分析](docs/PROJECT_ANALYSIS_CN.md)。

### 4. 参考必须经过物理执行

```mermaid
flowchart LR
  A[任务与阶段计划] --> B[轨迹程序]
  B --> C[运动学与结构检查]
  C -->|诊断反馈| B
  C -->|通过| D[电机驱动的真实物理执行]
  D -->|独立筛选通过| E[参考数据]
  E --> F[世界模型与BPTT训练]
  F --> G[真实物理评估]
```

通过运动学检查只说明提案满足对应的结构约束。它不能直接作为世界模型的真实状态转移。参考前端将提案与实际执行轨迹分开保存；只有物理筛选通过的数据才能导出为训练示范。

## 目录结构

```text
src/supermimic/
  wf/                     轮腿臂仿真、任务、模型、网络和训练
  mimicagent/             参考程序执行、检查、反馈和物理筛选
  world_model/            通用动力学网络与状态处理
  imagination/            想象展开、Actor目标和Critic目标
  kinematics/             可微正运动学
frameworks/fixed_base_arm/
  arm_smc/                六轴机械臂基线
  variants/supermimic_robust_control_v7/arm_smc/
                          结构化机械臂控制、规划和训练
configs/examples/         不含私有机器人参数的示例配置
examples/mimicagent/       轨迹程序与阶段计划示例
scripts/                  精选训练、数据处理与评估入口
tests/                    发布版测试
tools/                    资产检查、发布准备和内容审查
third_party/              上游来源与致谢
docs/                     项目分析、私有资产说明和仓库简介
```

本地研究目录中的历史实验和过程记录不属于 GitHub 发布内容。

## 环境与依赖

WF 路径使用 Python 3.11 或 3.12。已使用的本地 CPU 环境为 Python 3.11、PyTorch 2.7、MuJoCo 3.10、NumPy 1.26。CUDA 用于加速 Torch 学习；WF 参考后端的真实物理采集在 CPU 上执行。

```bash
git clone https://github.com/Robot-Nav/SuperMimic-Agent-WFArm.git
cd SuperMimic-Agent-WFArm
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

| 依赖 | 用途 | 安装范围 |
|---|---|---|
| PyTorch | 世界模型、策略、自动微分与优化 | 基础依赖 |
| MuJoCo | 真实动力学、接触检测和模型读取 | 基础依赖 |
| NumPy、SciPy | 数据处理、数值计算和旋转变换 | 基础依赖 |
| pytest | 测试 | `.[dev]` |
| OSQP、quadprog | 离线 QP 教师及约束诊断 | `.[teacher]` |
| imageio、imageio-ffmpeg、matplotlib | 可选录像和绘图 | `.[media]` |
| mjlab、MuJoCo Warp、Warp 及动作数据工具 | 保留的 G1 路径 | 单独 Python 3.12 环境中的 `.[g1]` |

使用 GPU 时先安装与显卡驱动匹配的 PyTorch。G1 额外依赖包含固定的 mjlab 提交及较新的仿真栈，不需要为运行 WF 基础代码安装它们。

## 运行步骤

### 1. 先运行不需要机器人模型的测试

```bash
python -m pytest -q tests/test_release_core.py
```

测试覆盖状态积分、Actor/Critic 目标、梯度传播、私有资产路径和发布边界，不把测试通过等同于机器人任务通过。

### 2. 接入有权使用的机器人资产

```bash
export WFARM_ASSET_DIR=/path/to/authorized/wf-arm-tbot
python tools/check_assets.py
```

资产目录包含 URDF、其引用的碰撞网格、关节元信息和 TCP 标定。固定底盘任务还需要私有工作限位配置。具体文件契约见[私有资产说明](docs/PRIVATE_ASSETS.md)。这些文件不应添加到 Git。

### 3. 检查轮腿臂训练链路

```bash
python -m supermimic.wf.train --smoke --device cpu --output outputs/wf_smoke
```

正式实验可以从 `configs/examples/wf_train.json` 复制配置，再调整数据、设备、训练预算及评估设置：

```bash
python -m supermimic.wf.train --config configs/examples/wf_train.json
```

示例只是实验起点，未附带已经训练好的策略。完整接续使用 `--resume`；`--warm-start` 转移的状态范围不同，不能混为同一种恢复方式。

### 4. 检查固定底盘六轴机械臂

在仓库根目录运行：

```bash
PYTHONPATH=src:frameworks/fixed_base_arm python -m arm_smc.train \
  --config configs/examples/fixed_arm_smoke.json
```

该配置生成小规模独立参考集并检查训练和评估流程。研究控制效果时需要增加预算，并保留独立验证和测试。结构化分支保留前序策略，可能需要本地训练的源检查点和目标编码器；仓库不分发这些权重。

### 5. 检查参考生成与训练接入

```bash
python scripts/wf_mimicagent_pilot.py \
  --output outputs/mimicagent_reference --terminal-feedback

python scripts/wf_mimicagent_smoke.py \
  --library outputs/mimicagent_reference/physical_library \
  --output outputs/mimicagent_training_check
```

末段反馈使用现有离线教师，属于本项目的参考生成适配。导出器校验执行数据和物理配置，并按独立请求保留训练/验证分割。教师轨迹可行不代表学生已经学会任务。

## 评估应记录什么

到达任务检查末尾连续保持的位姿和速度误差；跟踪任务检查请求轨迹中满足误差要求的帧比例。物理失败单独统计，失败后未执行的请求帧仍应留在跟踪分母中。最终测试不能参与世界模型、BC、Actor 更新或检查点选择。

世界模型误差下降、训练预算完成、参考筛选通过和最终策略成功分别代表不同结果。代码保留了动作响应检查、真实物理评估和回放边界处理，便于分析这些差异。底盘以 0.5–1.0 m/s 行进时的操作能力需要单独实验，不能从静态机械臂或 G1 动作跟踪结果推得。

## 协议、致谢与交流

### 致谢

- 感谢 **SuperMimic 作者团队**提出可微世界模型与 BPTT 的学习框架，为本项目的动力学建模和策略训练提供基础。
- 感谢 **MimicAgent 作者团队**提出提示词到参考轨迹、结构检查和诊断修正的流程，为本项目的参考生成前端提供思路。
- 感谢 **Yuxi Liu / CuboiLeo** 的 SuperMimic-Reproduced 实现，以及本项目使用的仿真、学习和数值计算库的维护者。

论文全名、作者和项目链接列于 [NOTICE](NOTICE.md)。

代码采用 [MIT 协议](LICENSE)，保留原有版权声明。上游来源和依赖致谢见 [NOTICE](NOTICE.md) 与 [third_party](third_party/README.md)。机器人资产和本地实验产物不在本仓库的软件授权范围内。

- [SuperMimic 原论文](https://arxiv.org/abs/2610.28480)
- [MimicAgent 原论文](https://arxiv.org/abs/2609.24145)
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)

欢迎通过 [Issues](https://github.com/Robot-Nav/SuperMimic-Agent-WFArm/issues) 交流训练现象、复现问题和控制方法。提交问题时附上代码版本、软件环境、配置及失败描述，请勿附带保密机器人文件或访问凭据。

[English](README.md) · [中文项目分析](docs/PROJECT_ANALYSIS_CN.md)
