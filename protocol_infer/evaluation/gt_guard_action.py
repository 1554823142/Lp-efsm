"""
模块二：GT guard/action 标注体系

从协议规范中提炼 guard 约束和 action 副作用，构成评估的参照答案（Ground Truth）。

数据结构
--------
FieldConstraint  : 单个字段的约束（等于某值 / 属于某范围 / 属于某集合）
GuardSpec        : 一条转移的所有字段约束组成的 guard 规范
ActionSpec       : 一条转移触发后应发生的变量变化规范
TransitionSpec   : 完整转移规范（src_type, dst_type, guard, action）
ProtocolGT       : 一个协议的全部 TransitionSpec

内置协议
--------
Modbus TCP（读线圈/离散/寄存器、写单个/多个寄存器、异常响应）
DNP3（初始化、轮询请求/响应、确认、不支持）
IEC 60870-5-104（STARTDT、STOPDT、TESTFR、I 帧数据、S 帧确认）

扩展
----
仿照 build_modbus_gt() 写构建函数，注册到 GT_BUILDERS 即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union


# ---------------------------------------------------------------------------
# 约束原语
# ---------------------------------------------------------------------------

@dataclass
class FieldConstraint:
    """
    对单个字段的约束。

    三种形式（互斥，优先级：eq > in_set > in_range）：
      eq:       字段值必须等于 value
      in_set:   字段值必须属于 allowed_values
      in_range: 字段值必须在 [lo, hi] 区间（闭区间）

    absent_ok:  若字段不在实际报文中，是否视为满足约束（默认 False）
    """
    field_name: str
    eq: Optional[Any] = None
    in_set: Optional[Set[Any]] = None
    in_range: Optional[tuple] = None   # (lo, hi)
    absent_ok: bool = False

    def check(self, fields: Dict[str, Any]) -> bool:
        if self.field_name not in fields:
            return self.absent_ok
        v = fields[self.field_name]
        if self.eq is not None:
            return v == self.eq
        if self.in_set is not None:
            return v in self.in_set
        if self.in_range is not None:
            lo, hi = self.in_range
            return lo <= v <= hi
        return True   # 无约束 -> 总满足

    def describe(self) -> str:
        if self.eq is not None:
            return f"{self.field_name}=={self.eq}"
        if self.in_set is not None:
            return f"{self.field_name} in {sorted(self.in_set)}"
        if self.in_range is not None:
            return f"{self.in_range[0]} <= {self.field_name} <= {self.in_range[1]}"
        return f"{self.field_name} exists"


# ---------------------------------------------------------------------------
# Guard 规范
# ---------------------------------------------------------------------------

@dataclass
class GuardSpec:
    """
    一条转移的 guard 规范：所有 constraints 同时满足时 guard 为真。
    同时支持自定义 predicate（优先级高于 constraints）。
    """
    constraints: List[FieldConstraint] = field(default_factory=list)
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    def check(self, fields: Dict[str, Any]) -> bool:
        if self.predicate is not None:
            return self.predicate(fields)
        return all(c.check(fields) for c in self.constraints)

    @property
    def field_names(self) -> Set[str]:
        return {c.field_name for c in self.constraints}

    def describe(self) -> str:
        if self.description:
            return self.description
        return " AND ".join(c.describe() for c in self.constraints) or "TRUE"


# ---------------------------------------------------------------------------
# Action 规范
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    """
    一条转移的 action 副作用规范。

    changed_vars:   转移后应被修改的变量集合（字段名）
    invariant_vars: 转移后应保持不变的变量集合
    computed:       {var_name: callable(fields) -> expected_value}
                    描述某个变量在执行 action 后的期望值
    description:    自然语言描述
    """
    changed_vars: Set[str] = field(default_factory=set)
    invariant_vars: Set[str] = field(default_factory=set)
    computed: Dict[str, Callable[[Dict[str, Any]], Any]] = field(default_factory=dict)
    description: str = ""

    def check_diff(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, bool]:
        """
        给定转移前后的字段快照，逐项验证 action 规范。
        返回 {变量名: 是否符合规范}。
        """
        results: Dict[str, bool] = {}
        for v in self.changed_vars:
            results[v] = (before.get(v) != after.get(v))
        for v in self.invariant_vars:
            results[v] = (before.get(v) == after.get(v))
        for v, fn in self.computed.items():
            expected = fn(before)
            results[v] = (after.get(v) == expected)
        return results


# ---------------------------------------------------------------------------
# 转移规范 & 协议 GT
# ---------------------------------------------------------------------------

@dataclass
class TransitionSpec:
    """
    完整转移规范。

    src_types:  触发此转移的源消息类型集合（与 msg_type / TraceRecord.msg_type 对齐）
    dst_types:  转移后的目标消息类型集合
    guard:      guard 规范
    action:     action 副作用规范
    label:      人类可读标签
    """
    src_types: Set[str]
    dst_types: Set[str]
    guard: GuardSpec = field(default_factory=GuardSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    label: str = ""


@dataclass
class ProtocolGT:
    """
    一个协议的完整 GT 规范。

    transitions:       全部转移规范列表
    guard_fields:      所有 guard 中出现的字段名（评估覆盖率时使用）
    action_vars:       所有 action 中应变化的变量集合（评估覆盖率时使用）
    """
    protocol: str
    transitions: List[TransitionSpec] = field(default_factory=list)

    @property
    def guard_fields(self) -> Set[str]:
        result: Set[str] = set()
        for ts in self.transitions:
            result.update(ts.guard.field_names)
        return result

    @property
    def action_vars(self) -> Set[str]:
        result: Set[str] = set()
        for ts in self.transitions:
            result.update(ts.action.changed_vars)
            result.update(ts.action.computed.keys())
        return result

    def find_matching_transitions(
        self,
        src_type: str,
        dst_type: str,
    ) -> List[TransitionSpec]:
        """查找匹配给定 src/dst 消息类型的转移规范。"""
        return [
            ts for ts in self.transitions
            if src_type in ts.src_types and dst_type in ts.dst_types
        ]


# ---------------------------------------------------------------------------
# 内置协议 GT 构建函数
# ---------------------------------------------------------------------------

def build_modbus_gt() -> ProtocolGT:
    """
    Modbus TCP GT 规范。

    消息类型命名与 ModbusTCPLabeler 输出对齐：
      fc_01_c2s, fc_03_s2c, ...
    """
    gt = ProtocolGT(protocol="MODBUS")

    # ---- 读线圈/离散输入请求 → 响应 (fc=01/02) ----
    for fc in (0x01, 0x02):
        fc_hex = f"{fc:02x}"
        gt.transitions.append(TransitionSpec(
            src_types={f"fc_{fc_hex}_c2s"},
            dst_types={f"fc_{fc_hex}_s2c"},
            guard=GuardSpec(
                constraints=[
                    FieldConstraint("fc", eq=fc),
                    FieldConstraint("proto_id", eq=0),
                    FieldConstraint("quantity", in_range=(1, 2000)),
                ],
                description=f"fc={fc_hex}, proto_id=0, 1<=quantity<=2000",
            ),
            action=ActionSpec(
                changed_vars={"byte_count"},
                invariant_vars={"txn_id", "unit_id"},
                computed={
                    "byte_count": lambda f, fc=fc: (f.get("quantity", 0) + 7) // 8
                },
                description="byte_count = ceil(quantity/8)",
            ),
            label=f"read_coils_fc{fc_hex}",
        ))

    # ---- 读保持/输入寄存器请求 → 响应 (fc=03/04) ----
    for fc in (0x03, 0x04):
        fc_hex = f"{fc:02x}"
        gt.transitions.append(TransitionSpec(
            src_types={f"fc_{fc_hex}_c2s"},
            dst_types={f"fc_{fc_hex}_s2c"},
            guard=GuardSpec(
                constraints=[
                    FieldConstraint("fc", eq=fc),
                    FieldConstraint("proto_id", eq=0),
                    FieldConstraint("quantity", in_range=(1, 125)),
                ],
                description=f"fc={fc_hex}, proto_id=0, 1<=quantity<=125",
            ),
            action=ActionSpec(
                changed_vars={"byte_count"},
                invariant_vars={"txn_id", "unit_id"},
                computed={
                    "byte_count": lambda f: f.get("quantity", 0) * 2
                },
                description="byte_count = quantity * 2",
            ),
            label=f"read_registers_fc{fc_hex}",
        ))

    # ---- 写单个寄存器 (fc=06) ----
    gt.transitions.append(TransitionSpec(
        src_types={"fc_06_c2s"},
        dst_types={"fc_06_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("fc", eq=0x06),
                FieldConstraint("proto_id", eq=0),
            ],
            description="fc=06, proto_id=0",
        ),
        action=ActionSpec(
            changed_vars=set(),
            invariant_vars={"txn_id", "unit_id", "start_addr"},
            description="echo request (start_addr/value mirrored)",
        ),
        label="write_single_register",
    ))

    # ---- 写多个寄存器请求 (fc=10 = 0x10) ----
    gt.transitions.append(TransitionSpec(
        src_types={"fc_10_c2s"},
        dst_types={"fc_10_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("fc", eq=0x10),
                FieldConstraint("proto_id", eq=0),
                FieldConstraint("quantity", in_range=(1, 123)),
            ],
            description="fc=0x10, proto_id=0, 1<=quantity<=123",
        ),
        action=ActionSpec(
            changed_vars=set(),
            invariant_vars={"txn_id", "unit_id", "start_addr", "quantity"},
            description="response echoes start_addr and quantity",
        ),
        label="write_multiple_registers",
    ))

    # ---- 异常响应 ----
    gt.transitions.append(TransitionSpec(
        src_types={f"fc_{fc:02x}_c2s" for fc in range(0x01, 0x80)},
        dst_types={f"fc_{0x80 | fc:02x}_s2c" for fc in range(0x01, 0x80)},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("proto_id", eq=0),
            ],
            predicate=lambda f: (f.get("fc", 0) & 0x80) != 0,
            description="fc & 0x80 != 0 (exception response)",
        ),
        action=ActionSpec(
            changed_vars={"exception_code", "is_exception"},
            description="exception_code set, is_exception=1",
        ),
        label="exception_response",
    ))

    return gt


def build_dnp3_gt() -> ProtocolGT:
    """
    DNP3 GT 规范。

    消息类型命名与 DNP3Labeler 输出对齐：
      func_0_prm1_c2s (初始化请求), func_0_prm0_s2c (确认) ...
    """
    gt = ProtocolGT(protocol="DNP3")

    # ---- 链路状态请求 (RESET_LINK, func=0, prm=1) -> 确认 (func=0, prm=0) ----
    gt.transitions.append(TransitionSpec(
        src_types={"func_0_prm1_c2s"},
        dst_types={"func_0_prm0_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("func", eq=0),
                FieldConstraint("prm", eq=1),
            ],
            description="func=0(RESET_LINK), prm=1",
        ),
        action=ActionSpec(
            changed_vars={"prm"},
            invariant_vars={"dest", "src"},
            description="prm flips 1->0 in ACK; src/dest swap",
        ),
        label="reset_link",
    ))

    # ---- 用户数据请求 (UNCONFIRMED_USER_DATA, func=4, prm=1) ----
    gt.transitions.append(TransitionSpec(
        src_types={"func_4_prm1_c2s"},
        dst_types={"func_0_prm0_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("func", eq=4),
                FieldConstraint("prm", eq=1),
            ],
            description="func=4(UNCONFIRMED_USER_DATA), prm=1",
        ),
        action=ActionSpec(
            changed_vars={"func", "prm"},
            invariant_vars={"dest", "src"},
            description="ACK with func=0, prm=0",
        ),
        label="unconfirmed_user_data",
    ))

    # ---- TEST_LINK_STATES 请求 (func=2, prm=1) -> 确认 ----
    gt.transitions.append(TransitionSpec(
        src_types={"func_2_prm1_c2s"},
        dst_types={"func_0_prm0_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("func", eq=2),
                FieldConstraint("prm", eq=1),
            ],
            description="func=2(TEST_LINK_STATES), prm=1",
        ),
        action=ActionSpec(
            changed_vars={"func", "prm"},
            invariant_vars={"dest", "src"},
            description="ACK with func=0, prm=0",
        ),
        label="test_link_states",
    ))

    # ---- 不支持的功能码 (func=f) ----
    gt.transitions.append(TransitionSpec(
        src_types={f"func_{f}_prm1_c2s" for f in range(0, 16)},
        dst_types={"func_f_prm0_s2c"},
        guard=GuardSpec(
            predicate=lambda f: f.get("prm", 0) == 1,
            description="prm=1 (primary), any unrecognized func -> NOT_SUPPORTED",
        ),
        action=ActionSpec(
            changed_vars={"func"},
            invariant_vars={"dest", "src"},
            description="func=0xF (NOT_SUPPORTED)",
        ),
        label="not_supported",
    ))

    return gt


def build_iec104_gt() -> ProtocolGT:
    """
    IEC 60870-5-104 GT 规范。

    消息类型命名与 IEC104Labeler 输出对齐：
      type_XX_c2s / type_XX_s2c
    """
    gt = ProtocolGT(protocol="IEC60870-104")

    # ---- STARTDT (控制域 U 格式) c2s -> s2c ----
    gt.transitions.append(TransitionSpec(
        src_types={"type_00_c2s"},  # apdu_len=4, U 帧 STARTDT_ACT
        dst_types={"type_00_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("frame_type", eq="U"),
                FieldConstraint("ctrl1", eq=0x07),   # STARTDT_ACT
            ],
            description="U-frame STARTDT_ACT (ctrl1=0x07)",
        ),
        action=ActionSpec(
            changed_vars={"ctrl1"},
            description="ctrl1 changes to 0x0B (STARTDT_CON)",
        ),
        label="startdt",
    ))

    # ---- STOPDT ----
    gt.transitions.append(TransitionSpec(
        src_types={"type_00_c2s"},
        dst_types={"type_00_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("frame_type", eq="U"),
                FieldConstraint("ctrl1", eq=0x13),   # STOPDT_ACT
            ],
            description="U-frame STOPDT_ACT (ctrl1=0x13)",
        ),
        action=ActionSpec(
            changed_vars={"ctrl1"},
            description="ctrl1 changes to 0x23 (STOPDT_CON)",
        ),
        label="stopdt",
    ))

    # ---- TESTFR ----
    gt.transitions.append(TransitionSpec(
        src_types={"type_00_c2s"},
        dst_types={"type_00_s2c"},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("frame_type", eq="U"),
                FieldConstraint("ctrl1", eq=0x43),   # TESTFR_ACT
            ],
            description="U-frame TESTFR_ACT (ctrl1=0x43)",
        ),
        action=ActionSpec(
            changed_vars={"ctrl1"},
            description="ctrl1 changes to 0x83 (TESTFR_CON)",
        ),
        label="testfr",
    ))

    # ---- I 帧：数据传输 (type_id=1..127 示意) ----
    gt.transitions.append(TransitionSpec(
        src_types={f"type_{tid:02x}_c2s" for tid in range(1, 128)},
        dst_types={f"type_{tid:02x}_s2c" for tid in range(1, 128)},
        guard=GuardSpec(
            constraints=[
                FieldConstraint("frame_type", eq="I"),
                FieldConstraint("type_id", in_range=(1, 127)),
                FieldConstraint("cause_of_tx", in_range=(1, 63)),
            ],
            description="I-frame, type_id in [1,127], cause_of_tx in [1,63]",
        ),
        action=ActionSpec(
            changed_vars={"ns", "nr"},
            description="ns increments by 1 in sender; nr echoes peer's ns",
        ),
        label="i_frame_data",
    ))

    # ---- S 帧：监督帧确认 ----
    gt.transitions.append(TransitionSpec(
        src_types={f"type_{tid:02x}_s2c" for tid in range(1, 128)},
        dst_types={"type_00_c2s"},   # S 帧不含 type_id，用 type_00 近似
        guard=GuardSpec(
            constraints=[
                FieldConstraint("frame_type", eq="S"),
            ],
            description="S-frame (supervisory)",
        ),
        action=ActionSpec(
            changed_vars={"nr"},
            invariant_vars={"common_addr"},
            description="nr = ns of last accepted I-frame",
        ),
        label="s_frame_supervisory",
    ))

    return gt


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

GT_BUILDERS: Dict[str, Callable[[], ProtocolGT]] = {
    "MODBUS": build_modbus_gt,
    "MODBUSTCP": build_modbus_gt,
    "DNP3": build_dnp3_gt,
    "IEC104": build_iec104_gt,
    "IEC60870-104": build_iec104_gt,
}


def get_gt(protocol: str) -> Optional[ProtocolGT]:
    """
    获取协议 GT 规范。未知协议返回 None。
    """
    builder = GT_BUILDERS.get(protocol.upper())
    if builder is None:
        return None
    return builder()


# ---------------------------------------------------------------------------
# 字节位变量名 → 语义字段名 别名映射表
#
# FeatureProcessor.extract_vars() 产生的变量名为 b{pos}（字节位置），
# GT 规范使用协议语义字段名。此表用于在评估时归一化两边的字段名。
# 注意：多字节字段（如 proto_id, quantity）会有多个字节位映射到同一语义名。
# ---------------------------------------------------------------------------

# 别名值可以是 str（单一语义）或 List[str]（同一字节位在不同方向有不同语义）
PROTOCOL_VAR_ALIASES: Dict[str, Dict[str, Union[str, List[str]]]] = {
    "MODBUS": {
        # MBAP Header（字节位变量名）
        "b0": "txn_id",   "b1": "txn_id",
        "b2": "proto_id", "b3": "proto_id",
        "b4": "length",   "b5": "length",
        "b6": "unit_id",
        "b7": "fc",
        # PDU（请求时 b8/b9=start_addr，响应时 b8=byte_count；方向无感知时两者均计）
        "b8": ["start_addr", "byte_count"],
        "b9": "start_addr",
        "b10": "quantity",    "b11": "quantity",
        # 静态项变量名（s{byte_pos}），与对应字节位同义
        "s0": "txn_id",
        "s2": "proto_id",  "s3": "proto_id",
        "s4": "length",
        "s6": "unit_id",
    },
    "DNP3": {
        "b2": "length",
        "b3": "ctrl",
        "b4": "dest",  "b5": "dest",
        "b6": "src",   "b7": "src",
    },
    "IEC104": {
        "b1": "apdu_len",
        "b2": "ctrl1",
        "b3": "ctrl2",
        "b6": "type_id",
        "b7": "vsq",
    },
}


def normalize_var_names(var_names: Set[str], protocol: str) -> Set[str]:
    """
    将字节位变量名集合（如 {b7, b6}）通过别名表转换为语义字段名集合（如 {fc, unit_id}）。
    别名值支持 str（单一语义）或 List[str]（多语义，如 b8 在请求/响应中含义不同）。
    未在表中的变量名保持不变。
    """
    aliases = PROTOCOL_VAR_ALIASES.get(protocol.upper(), {})
    normalized: Set[str] = set()
    for name in var_names:
        mapped = aliases.get(name, name)
        if isinstance(mapped, list):
            normalized.update(mapped)
        else:
            normalized.add(mapped)
    return normalized
