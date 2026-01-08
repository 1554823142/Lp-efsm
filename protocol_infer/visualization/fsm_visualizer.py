# fsm_visualizer.py
import json
import graphviz
import pandas as pd
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from enum import Enum

from protocol_infer.core.model.fsm import Transition

class FSMFormat(Enum):
    """支持的输出格式"""
    GRAPHVIZ = "graphviz"     # Graphviz DOT格式
    JSON = "json"            # JSON格式（前端友好）
    CSV = "csv"              # CSV格式（便于分析）
    PLANTUML = "plantuml"    # PlantUML格式
    Mermaid = "mermaid"      # Mermaid格式（前端图表库）

class FSMVisualizer:
    """FSM可视化器"""
    
    def __init__(self, fsm, title: str = "Protocol State Machine"):
        """
        初始化FSM可视化器
        
        Args:
            fsm: FSM对象
            title: 图表的标题
        """
        self.fsm = fsm
        self.title = title
        self._node_colors = {
            'start': '#90EE90',      # 浅绿色
            'end': '#FFB6C1',        # 浅粉色
            'normal': '#87CEEB',     # 浅蓝色
            'highlight': '#FFD700'   # 金色（高亮）
        }
    
    def _get_state_color(self, state_id: int) -> str:
        """获取状态节点的颜色"""
        state = self.fsm.states[state_id]
        if state.is_start:
            return self._node_colors['start']
        elif state.is_end:
            return self._node_colors['end']
        else:
            return self._node_colors['normal']
    
    def _get_state_label(self, state_id: int) -> str:
        """获取状态节点的标签"""
        state = self.fsm.states[state_id]
        label = f"{state.name}"
        
        # 添加统计信息
        stats = []
        if state.visit_count > 0:
            stats.append(f"访问: {state.visit_count}")
        if state.hasNo is not None:
            stats.append(f"哈希: {state.hasNo}")
        
        if stats:
            label += f"\\n({', '.join(stats)})"
        
        # 添加特殊标记
        if state.is_start:
            label += "\\n[START]"
        if state.is_end:
            label += "\\n[END]"
            
        return label
    
    def _get_transition_label(self, transition: Transition) -> str:
        """获取转移边的标签"""
        label = transition.symbol
        
        # 添加额外信息
        extra_info = []
        if transition.prob is not None:
            extra_info.append(f"P={transition.prob:.2f}")
        if transition.output:
            extra_info.append(f"→{transition.output}")
        
        if extra_info:
            label += f"\\n({', '.join(extra_info)})"
        
        return label
    

    def to_graphviz(self, 
                   format: str = "png", 
                   filename: Optional[str] = None,
                   highlight_states: Optional[Set[int]] = None,
                   highlight_transitions: Optional[Set[int]] = None) -> graphviz.Digraph:
        """
        生成Graphviz图
        
        Args:
            format: 输出格式（png, svg, pdf等）
            filename: 输出文件名（不包含扩展名）
            highlight_states: 需要高亮的状态ID集合
            highlight_transitions: 需要高亮的转移ID集合
            
        Returns:
            graphviz.Digraph对象
        """
        if highlight_states is None:
            highlight_states = set()
        if highlight_transitions is None:
            highlight_transitions = set()
        
        # 创建有向图
        dot = graphviz.Digraph(
            name=self.title,
            format=format,
            graph_attr={
                'label': self.title,
                'labelloc': 't',
                'fontname': 'Arial',
                'fontsize': '16',
                'rankdir': 'LR',  # 从左到右布局
                'nodesep': '0.5',
                'ranksep': '1.0'
            },
            node_attr={
                'shape': 'circle',
                'style': 'filled',
                'fontname': 'Arial',
                'fontsize': '10'
            },
            edge_attr={
                'fontname': 'Arial',
                'fontsize': '9',
                'arrowsize': '0.7'
            }
        )
        
        # 添加状态节点
        for state_id, state in self.fsm.states.items():
            node_attrs = {
                'label': self._get_state_label(state_id),
                'fillcolor': self._get_state_color(state_id),
                'color': 'black',
                'penwidth': '1.0'
            }
            
            # 高亮处理
            if state_id in highlight_states:
                node_attrs.update({
                    'fillcolor': self._node_colors['highlight'],
                    'penwidth': '3.0',
                    'style': 'filled,bold'
                })
            
            # 起始状态特殊形状
            if state.is_start:
                node_attrs['shape'] = 'doublecircle'
                node_attrs['peripheries'] = '2'  # 双圈
            elif state.is_end:
                node_attrs['shape'] = 'doublecircle'
            
            dot.node(str(state_id), **node_attrs)
        
        # 添加转移边
        for transition in self.fsm.transitions:
            edge_attrs = {
                'label': self._get_transition_label(transition),
                'color': 'black' if transition.id not in highlight_transitions else 'red',
                'penwidth': '1.5' if transition.id not in highlight_transitions else '3.0'
            }
            
            # 检查是否有多个相同src->dst的转移（需要合并显示）
            same_transitions = [
                t for t in self.fsm.transitions 
                if t.src == transition.src and t.dst == transition.dst
            ]
            
            if len(same_transitions) > 1:
                # 合并多个转移为一个边，标签用逗号分隔
                labels = [self._get_transition_label(t) for t in same_transitions]
                edge_attrs['label'] = ',\\n'.join(labels)
                # 只添加一次
                if transition == same_transitions[0]:
                    dot.edge(str(transition.src), str(transition.dst), **edge_attrs)
            else:
                dot.edge(str(transition.src), str(transition.dst), **edge_attrs)
        
        # 添加不可达节点检测
        reachable_states = self._find_reachable_states()
        unreachable_states = set(self.fsm.states.keys()) - reachable_states
        if unreachable_states:
            dot.attr(label=f"{self.title}\\n(红色: {len(unreachable_states)}个不可达状态)", 
                    fontcolor='red', fontsize='12')
            for state_id in unreachable_states:
                dot.node(str(state_id), fillcolor='#FF9999', color='red')
        
        # 保存文件
        if filename:
            output_path = Path(filename)
            dot.render(filename=str(output_path.with_suffix('')), 
                      cleanup=True, 
                      format=format)
            print(f"Graphviz图已保存到: {output_path.with_suffix('.' + format)}")
        
        return dot
    
    def _find_reachable_states(self) -> Set[int]:
        """查找从起始状态可达的所有状态"""
        if self.fsm.start_state is None:
            return set()
        
        reachable = set()
        stack = [self.fsm.start_state]
        
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            
            # 获取所有出边
            current_state = self.fsm.states[current]
            for symbol, next_state in current_state.next_states.items():
                # 找到对应的状态ID
                for state_id, state_obj in self.fsm.states.items():
                    if state_obj is next_state:
                        stack.append(state_id)
                        break
        
        return reachable
    
    def generate_report(self) -> Dict[str, Any]:
        """
        生成FSM统计分析报告
        
        Returns:
            包含各种统计指标的字典
        """
        # 基本统计
        total_states = len(self.fsm.states)
        total_transitions = len(self.fsm.transitions)
        
        # 状态类型统计
        start_states = sum(1 for s in self.fsm.states.values() if s.is_start)
        end_states = sum(1 for s in self.fsm.states.values() if s.is_end)
        normal_states = total_states - start_states - end_states
        
        # 转移统计
        unique_symbols = set(t.symbol for t in self.fsm.transitions)
        
        # 计算平均度数
        total_in_degree = sum(len(s.prev_states) for s in self.fsm.states.values())
        total_out_degree = sum(len(s.next_states) for s in self.fsm.states.values())
        avg_in_degree = total_in_degree / total_states if total_states > 0 else 0
        avg_out_degree = total_out_degree / total_states if total_states > 0 else 0
        
        # 路径分析
        reachable_states = self._find_reachable_states()
        unreachable_states = set(self.fsm.states.keys()) - reachable_states
        
        # 访问频率统计
        visit_counts = [s.visit_count for s in self.fsm.states.values()]
        
        report = {
            "基本信息": {
                "标题": self.title,
                "状态总数": total_states,
                "转移总数": total_transitions,
                "唯一符号数": len(unique_symbols),
                "起始状态": self.fsm.start_state,
                "生成时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "状态统计": {
                "起始状态数": start_states,
                "结束状态数": end_states,
                "普通状态数": normal_states,
                "可达状态数": len(reachable_states),
                "不可达状态数": len(unreachable_states)
            },
            "图论指标": {
                "平均入度": round(avg_in_degree, 2),
                "平均出度": round(avg_out_degree, 2),
                "最大入度": max(len(s.prev_states) for s in self.fsm.states.values()),
                "最大出度": max(len(s.next_states) for s in self.fsm.states.values()),
                "图密度": round(total_transitions / (total_states * (total_states - 1)), 4)
            },
            "访问统计": {
                "总访问次数": sum(visit_counts),
                "平均访问次数": round(sum(visit_counts) / total_states, 2) if total_states > 0 else 0,
                "最大访问次数": max(visit_counts) if visit_counts else 0,
                "最小访问次数": min(visit_counts) if visit_counts else 0
            },
            "符号频率": self._get_symbol_frequency(),
            "不可达状态列表": list(unreachable_states) if unreachable_states else []
        }
        
        return report
    
    def _get_symbol_frequency(self) -> Dict[str, int]:
        """统计符号出现频率"""
        symbol_freq = {}
        for transition in self.fsm.transitions:
            symbol_freq[transition.symbol] = symbol_freq.get(transition.symbol, 0) + 1
        
        # 按频率排序
        sorted_freq = dict(sorted(
            symbol_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_freq
    
    def print_report(self, output_file: Optional[str] = None) -> None:
        """
        打印或保存报告
        
        Args:
            output_file: 输出文件路径（可选）
        """
        report = self.generate_report()
        
        # 控制台输出
        print("=" * 60)
        print(f"FSM分析报告: {self.title}")
        print("=" * 60)
        
        for section, data in report.items():
            print(f"\n{section}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")
        
        print("\n" + "=" * 60)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(report, f, allow_unicode=True, default_flow_style=False)
            print(f"报告已保存到: {output_file}")