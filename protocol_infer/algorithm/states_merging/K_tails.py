from protocol_infer.core.algorithm.state_merge import StateMerger
from protocol_infer.core.model.fsm import FSM

class KTailStateMerger(StateMerger):

    def __init__(self, k: int):
        self.k = k              # nominate "k" value
        

    def merge(self, fsm: FSM) -> FSM:
        '''
            如果两个状态在未来的k步（k步之后）具有完全相同的行为，
            那么这两个状态可以合并
        '''

        # 迭代合并，直到没有状态可以合并
        while True:
            initial_state_count = len(fsm.states)

            # 1.计算每个状态的sig
            sig_map = {}        # sid->sig
            for sid in fsm.states:
                sig_map[sid] = self.signiture_compute(sid, self.k, fsm)

            # 2.按照sig分桶
            group_buckets = {}      # sig->List[sid]
            for sid, sig in sig_map.items():
                group_buckets.setdefault(sig, []).append(sid)

            # 3.合并每一组
            merged_any = False
            for group in group_buckets.values():
                if len(group) > 1:
                    self.merge_group(group, fsm)
                    merged_any = True

            fsm.remove_duplicate_transitions()      # 将k-tail合并后的冗余相同转移边去除

            if not merged_any or len(fsm.states) == initial_state_count:
                break
        
        return fsm

    def signiture_compute(self, sid: int, k: int, fsm: FSM) ->tuple:
        '''
            计算状态的 特征签名
        '''
        state = fsm.states[sid]                    # 当前状态

        if k == 0:      # 基准情况, 表示终止与非终止不会合并
            return ("END" if state.is_end else "NONEND",)

        sig = []
        # 按符号排序, 确保签名的顺序是确定的, 使得相同的状态行为产生完全相同的签名
        for symbol, next_sid in sorted(state.next_states.items()):  
            sig.append(
                (symbol, self.signiture_compute(next_sid, k - 1, fsm))
            )

        return tuple(sig)       # 返回结果就是当前状态为根, 深度为k的行为树的序列化表示


    def merge_group(self, group, fsm):
        merged_state = group[0]

        # 将剩下的状态合并
        for sid in group[1:]:
            fsm.merge_two_state(merged_state, sid)
    

    
