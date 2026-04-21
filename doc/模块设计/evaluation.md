# 评价指标

指标主要集中与这几个维度进行评估系统:

- 端到端行为可重放能力（Replayability）

- 状态机结构正确性（Structural Fidelity）

- 约束规则语义准确性（Constraint Correctness）

不采用传统的"是否到达终态"的二值评估方法:

流量捕获/分段不完整, 有可能不是完整的起始--终止态, 并且如果一个guard出现误判, 则会导致整个会话被拒绝, 要求过于严格, 所以采用粗略整体, 再考虑局部, 并允许部分断链的评估思想进行评估

## 端到端重放与结构匹配指标 (Replay & Behavioral Metrics)

此类指标通过在测试集上按步执行“回放（Replay）”，验证推断出的模型对真实流量的在线跟踪与接受能力。

匹配一共分为三种层级:(也可以理解为不同等级的严格标准)

| 类型                        | 条件                           | 含义               |
| --------------------------- | ------------------------------ | ------------------ |
| Replay Match                | 能找到任意可执行边             | 能解释行为（宽松） |
| **State Match（结构匹配）** | 当前状态可直接转移（无重同步） | **状态机结构正确** |
| Strict Match                | 状态 + Guard 均正确            | 完全正确           |

### Session Full Match Rate (会话平均匹配率)
- **含义**：衡量“状态机结构对会话主流程的贴合程度”。对每个会话，严格统计能在**当前状态合法找到后续转移的“步数占比”**（不计入全局重同步），然后对所有会话取**平均值**。

  解释**模型在不同会话上是否稳定**

- **代码实现位置**：
  
  - **类/方法**：`protocol_infer/visualization/replay.py` -> `summarize_replay_by_session()` 方法中计算。
  - **字段名**：`session_state_step_match_rate`
  
- 计算流程:

  - 所有回放步骤按session分桶
  - 逐个session计算结构匹配率(统计该session总步数, 统计结构匹配的步数(需要保证不是Replay Match), 计算匹配率)
  - 对所有的session取平均

  伪代码:

  ```python
  for each session:
      state_ok = sum(结构匹配步数)
      rate = state_ok / len(session_steps)
      收集 rate
  
  最终结果 = 平均(rate)
  ```

### Message Match Accuracy (消息级结构匹配率)
- **含义**：**消息级别**的**整体匹配准确率**。计算方式为 `(全体消息步里：严格在当前状态匹配到转移的步数) / (总步数)`。

  反映**模型整体能解释多少行为**

- **代码实现位置**：
  
  - **类/方法**：`protocol_infer/visualization/replay.py` -> `summarize_replay_by_session()` 方法中计算。
  - **字段名**：`step_state_replay_accuracy`
  
- 计算流程:

  ```python
  steps_state = sum(结构匹配步数)
  steps_total = 总步数
  
  accuracy = steps_state / steps_total
  ```

### Steps Resynced (重同步步数)
- **含义**：回放中发生“当前状态无路可走，被迫全局搜索图中其他同 Symbol 边进行重同步”的次数。
- **代码实现位置**：
  - **类/方法**：`protocol_infer/visualization/replay.py` -> `summarize_replay_by_session()` 方法中计算。
  - **字段名**：`steps_resynced`

### Sessions Evaluated / Steps Total (评估样本量)
- **含义**：参与回放统计的会话数量与总消息步数。
- **代码实现位置**：
  - **类/方法**：`protocol_infer/visualization/replay.py` -> `summarize_replay_by_session()` 方法中计算。
  - **字段名**：`sessions_replay_evaluated` 和 `steps_total`

---

## 2. 约束规则辅助指标 (Guard PRF)

此类指标需要预先定义好协议的真实规范（Ground Truth），主要用于辅助判断学习到的约束规则是否符合规范语义。

### 2.1 Guard Precision / Guard Recall / Guard F1
- **含义**：将推断出的 Guard 字段集（推断模型中用于分支判断的变量集合）与真实规范字段集做交并比计算，得到字段级别的精确率、召回率与 F1 值。
- **代码实现位置**：
  - **类/方法**：`protocol_infer/evaluation/efsm_evaluator.py` -> `GuardFieldEvaluator.evaluate()` 方法。
  - **字段名**：`guard_precision`, `guard_recall`, `guard_f1`

---

*注：上述所有前端展示指标的聚合与最终接口返回逻辑，均由 `protocol_infer/visualization/service.py` 中的 `VisualizationService._build_metrics()` 方法统一负责处理映射与封装。*
