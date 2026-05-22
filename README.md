# P-EFSM: Probabilistic Extended Finite State Machine Inference for Industrial Protocol Reverse Engineering

从 PCAP 网络流量中自动推断工业协议的**概率扩展有限状态机（P-EFSM）**，支持模型训练、多维评估与 Web 可视化交互分析。

## Overview

P-EFSM 是一个面向工业协议逆向工程的**分层推断系统**。它从原始 PCAP 文件出发，经过四个层次的处理，逐步构建出带有**概率转移**、**守卫条件（Guard）** 和**动作函数（Action）** 的扩展有限状态机。

系统支持 **Modbus、S7COMM、IEC60870-104** 工业协议，并提供了 Web 可视化交互方式。

## Architecture

系统采用**四层流水线架构**，层间通过 Core 定义的数据模型与接口通信：

```
PCAP File
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 0 & 1: pcap_layer            │  PCAP parsing → 会话聚合 → 事件提取 → Trace
│  (PCAPPipeline)                     │
└─────────────────────────────────────┘
    │ Trace (events + session contexts)
    ▼
┌─────────────────────────────────────┐
│  Layer 2: control_flow_layer        │  特征提取 → 聚类抽象 → PTA 构建 → K-tail 合并 → FSM
│  (ControlFlowPipeline)              │
└─────────────────────────────────────┘
    │ FSM
    ▼
┌─────────────────────────────────────┐
│  Layer 3: data_flow_layer           │  变量提取 → Guard/Action 学习 → EFSM
│  (DataFlowPipeline)                 │
└─────────────────────────────────────┘
    │ EFSM
    ▼
┌─────────────────────────────────────┐
│  Layer 4: probabilistic_layer       │  训练回放统计 → 转移概率 → 置信等级 → P-EFSM
│  (ProbabilisticPipeline)            │
└─────────────────────────────────────┘
    │ P-EFSM
    ▼
    Evaluation & Visualization
```

### Core

核心抽象层，定义全系统统一的数据结构与算法接口：

- **datamodel** — `RawPacket`, `Session`, `MessageEvent`, `Trace`, `AbstractMessage`, `SessionContext` 等跨层数据结构
- **model** — `FSM`, `EFSM`, `PEFSM` 状态机模型
- **interface** — `PCAPParser`, `FlowBuilder`, `FSMInfer`, `EFSMBuilder`, `ProbTrainer` 等策略接口
- **algorithm** — 聚类算法（KMeans、层次聚类、DBSCAN、规则聚类）、状态合并（K-tail）、Guard/Action 学习

### pcap_layer

数据预处理层：

- **parser** — 基于 Scapy 解析 PCAP 文件 → `RawPacket`
- **session** — 五元组会话聚合，方向判定（端口优先级 / 首包 src_ip）
- **segmentation** — 报文级别分割与 TCP 重组

### control_flow_layer

控制流构建层（FSM 推断）：

- **Feature Extraction** — 支持传统统计特征（报文长度、端口、方向）与 **Apriori 伪字段发现**
- **Clustering** — KMeans / 层次聚类 / DBSCAN / 规则聚类，将特征向量映射为离散符号
- **FSM Inference** — 基于 **PTA（Prefix Tree Acceptor）** 构建状态机
- **State Merging** — **K-tail** 算法合并冗余状态，宁留冗余不误删

### data_flow_layer

数据流构建层（EFSM 推断）：

- **Session Context** — 统计会话特征（包比例、字节比例、请求-响应配对率、时序特征等）
- **Variable Extraction** — 从报文 payload 中提取字段变量
- **Guard Learning** — 基于**改进 Apriori 算法**学习单变量约束与多变量关联规则，支持常量/离散值/序列值/连续值类型推断
- **Action Learning** — 按变量类型确定更新策略（keep / delta）
- **Field Boundary Detection** — 固定偏移 + 多字节解析，适配工控协议字段特点（[`protocol_infer/algorithm/field_detection/`](protocol_infer/algorithm/field_detection/)）

### probabilistic_layer

概率层：

- 基于训练会话在 EFSM 上的重放统计，为每条转移计算：
  - **traverse_count** — 观测次数
  - **P(transition | src)** — 状态出边条件概率
  - **confidence** — high / medium / low 置信等级
- 支持按概率或访问次数对低置信转移进行**剪枝**，保留主路径可达性

## Supported Protocols

| Protocol | Port | GT Guard Spec | Directory |
|----------|------|---------------|-----------|
| MODBUS | 502 | ✅ | [`Data/MODBUS/`](Data/MODBUS/) |
| DNP3 | 20000 | ✅ | [`Data/DNP3/`](Data/DNP3/) |
| IEC104 | 2404 | ✅ | [`Data/IEC60870-104/`](Data/IEC60870-104/) |
| S7COMM | 102 | ❌ | [`Data/S7Comm/`](Data/S7Comm/) |
| EtherNet/IP | 44818 | ❌ | [`Data/Ethernet_IP/`](Data/Ethernet_IP/) |
| MQTT | 1883 | ❌ | [`Data/MQTT/`](Data/MQTT/) |

协议规范级 Ground Truth（GT）定义在 [`protocol_infer/evaluation/gt_guard_action.py`](protocol_infer/evaluation/gt_guard_action.py)。

## Evaluation

系统采用**多维评价体系**，避免传统"是否到达终态"二值指标的失真问题：

### Core Metrics (端到端重放)

| Metric | Description |
|--------|-------------|
| `session_state_step_match_rate` | 会话平均匹配率：每个会话内结构匹配步数占比的平均值 |
| `step_state_replay_accuracy` | 消息级结构匹配率：全体消息步中状态匹配的比例 |
| `steps_resynced` | 重同步步数：衡量结构漂移程度 |

### Reference Metrics (Guard / Action 语义)

| Metric | Description |
|--------|-------------|
| `guard_precision / recall / f1` | Guard 字段级精确率、召回率、F1 |
| `guard_violation_rate` | Guard 违规率 |
| `action_coverage_jaccard` | Action 变量覆盖的 Jaccard 相似度 |
| `state_diff_accuracy` | 状态差分准确率 |

### Sample Results

| Protocol | Session Match Rate | Message Match Rate | Guard F1 | Resync Steps |
|----------|:---:|:---:|:---:|:---:|
| MODBUS | 0.9902 | 0.9503 | 0.2353 | 8 |
| DNP3 | 0.9902 | 0.9357 | 0.0000 | 73 |
| IEC104 | 0.8884 | 0.8807 | 0.0930 | 21 |

## Web Visualization

基于 **FastAPI** 的 Web 可视化系统，集成模型训练、评估与交互分析。

### Quick Start

```bash
python -m uvicorn protocol_infer.visualization.webapp:app --host 127.0.0.1 --port 8010
```

打开浏览器访问 `http://127.0.0.1:8010`

### Core APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web 可视化主页 |
| `/api/health` | GET | 服务健康检查 |
| `/api/datasets` | GET | 获取可用数据集列表 |
| `/api/learn` | POST | 训练 P-EFSM 模型（支持 pcap / synthetic / pcap+synthetic 模式） |
| `/api/artifacts/{artifact_id}` | GET | 获取训练产物（模型 + 回放 + 指标） |
| `/api/artifacts/{artifact_id}/upload-pcap` | POST | 上传新 PCAP，基于已有模型进行回放分析 |

### Training Profiles

系统提供三种预设训练配置：

- **fast** — 4 PCAP / 120 sessions，适合快速验证
- **balanced** — 8 PCAP / 300 sessions，默认折中
- **thorough** — 16 PCAP / 800 sessions，覆盖优先

### Command Line Evaluation

```bash
python -m protocol_infer.evaluation.run_evaluation \
    --protocol MODBUS \
    --data-dir Data/MODBUS \
    --max-pcaps 6 --max-sessions 200 --test-ratio 0.2 --seed 42
```

## Project Structure

```
protocol_infer/
├── core/                           # 核心数据模型与接口
│   ├── datamodel/                  # RawPacket, Session, MessageEvent, Trace, ...
│   ├── model/                      # FSM, EFSM, PEFSM
│   ├── interface/                  # 各层策略接口
│   └── algorithm/                  # 聚类、状态合并、Guard/Action 算法
├── pcap_layer/                     # Layer 0 & 1: PCAP 解析与预处理
│   ├── parser/                     # Scapy 解析器
│   ├── session/                    # 五元组会话构建
│   ├── segmentation/               # 报文分割 / TCP 重组
│   └── pipeline.py                 # PCAP → Trace
├── control_flow_layer/             # Layer 2: 控制流推断 (FSM)
│   ├── features/                   # 特征提取 (含 Apriori 特征)
│   ├── abstraction/                # 聚类抽象
│   ├── inference/                  # PTA 构建
│   └── pipeline.py
├── data_flow_layer/                # Layer 3: 数据流推断 (EFSM)
│   ├── abstraction/                # 消息抽象
│   ├── feature/                    # 会话上下文特征
│   ├── trace_processor/            # 上下文管理
│   ├── inference/                  # EFSM 推断
│   └── pipeline.py
├── probabilistic_layer/            # Layer 4: 概率层 (P-EFSM)
│   ├── inference/                  # 概率统计
│   └── pipeline.py
├── algorithm/                      # 通用算法模块
│   ├── clustering/                 # KMeans, DBSCAN, 层次聚类, 规则聚类
│   ├── states_merging/             # K-tail
│   ├── guard_action/               # Apriori Guard / Interval Delta / Cross-message
│   └── field_detection/            # 动态字段检测
├── apriori/                        # Apriori 频繁项集挖掘
│   ├── core.py                     # AprioriCore 算法
│   └── miners.py                   # StaticFieldMiner, BytePositionTransactionBuilder
├── evaluation/                     # 评估系统
│   ├── run_evaluation.py           # 完整评估入口
│   ├── efsm_evaluator.py           # EnhancedEFSMEvaluator
│   ├── gt_guard_action.py          # 协议规范 GT
│   ├── supervised_eval.py          # 监督标签器
│   ├── field_extractor.py          # 字段提取器
│   └── metrics/                    # FSM / EFSM / 聚类指标
├── visualization/                  # Web 可视化
│   ├── webapp.py                   # FastAPI 应用
│   ├── service.py                  # VisualizationService
│   ├── serializer.py               # PEFSMSerializer
│   ├── replay.py                   # ReplayBuilder
│   └── static/                     # 前端页面
│       ├── index.html
│       ├── app.js
│       └── styles.css
└── pipline.py                      # 统一流水线入口
Data/                               # PCAP 数据集 (MODBUS, DNP3, IEC104, ...)
```

## Apriori 特征设计

系统在控制流层引入**基于 Apriori 算法的伪字段发现机制**，解决传统统计特征（报文长度、方向）无法区分仅关键字节不同的消息类型的问题：

1. **BytePositionTransactionBuilder** — 将消息字节编码为 `(偏移, 值)` 事务项集
2. **AprioriCore** — 逐层挖掘频繁项集，利用剪枝性质避免指数爆炸
3. **StaticFieldInterpreter** — 过滤全局静态字段，提取最大频繁项集作为消息类型指纹
4. **AprioriFeatureExtraction** — 将偏移值、项集匹配 one-hot、归一化长度、方向拼接为特征向量

对比实验表明，Apriori 特征在 ARI / NMI / V-measure 等聚类指标上显著优于基线特征。

## Dependencies

- Python 3.10+
- FastAPI + Uvicorn
- Scapy (PCAP parsing)
- NumPy / SciPy (KMeans, 层次聚类, DBSCAN)
- scikit-learn (聚类指标)
- matplotlib + graphviz (可视化布局)
