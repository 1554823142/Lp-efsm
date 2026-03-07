from typing import List, Tuple, FrozenSet, Dict, Any

class AprioriCore:
    def frequent_itemsets(self, transactions: List[FrozenSet[Any]], min_support: float) -> List[Tuple[FrozenSet[Any], float]]:
        n = len(transactions)
        if n == 0:
            return []
        item_counts: Dict[FrozenSet[Any], int] = {}
        singles: Dict[FrozenSet[Any], int] = {}

        # 1. 统计单项集的支持度(K = 1)
        for t in transactions:
            seen = set()
            for it in t:
                if it in seen:
                    continue
                s = frozenset([it])
                singles[s] = singles.get(s, 0) + 1  # 统计每个单项出现的次数
                seen.add(it)
        L: List[Dict[FrozenSet[Any], int]] = []         # 按层次的频繁项集列表, 仅保留满足支持度的
        L.append({k: v for k, v in singles.items() if v / n >= min_support})

        # 2. 不断扩大集合大小, 增加k
        k = 2
        while L[-1]:            # 上一层有频繁项集才继续扩展(也就是看上一层是否有元素)
            prev = list(L[-1].keys())
            prev_sorted = [tuple(sorted(x)) for x in prev]  # 首先排序, 确保后续合并时可以前缀比较
            candidates: Dict[FrozenSet[Any], int] = {}
            m = len(prev_sorted)
            for i in range(m):
                for j in range(i + 1, m):
                    a = prev_sorted[i]
                    b = prev_sorted[j]
                    '''
                    Apriori 连接规则：把两个频繁 (k-1) 项集合并为候选 k 项集的条件是
                    前 k-1 个元素完全相同，只有最后一个元素不同, 也就是增量为一的合并
                    '''
                    if a[:-1] == b[:-1]:
                        c = frozenset(a + (b[-1],))

                        # 剪枝: 检查c的所有(k-1)子集是否都是频繁项集
                        if self._all_subsets_frequent(c, L[-1]):
                            candidates[c] = 0       # 只有频繁集才参与计数
            for t in transactions:
                for c in list(candidates.keys()):
                    if c.issubset(t):       # 候选集是否包含于此事务
                        candidates[c] += 1  # 计数

            # 支持度: v / n, 低于阈值直接丢弃, 即丢弃不频繁的
            L.append({k: v for k, v in candidates.items() if v / n >= min_support})
            k += 1

        # 3. 收集结果(所有层)
        all_fis: List[Tuple[FrozenSet[Any], float]] = []
        for level in L:
            for k, v in level.items():
                all_fis.append((k, v / n))
                item_counts[k] = v      # 后计算置信度需要
        return all_fis

    def association_rules(self, frequent_itemsets: List[Tuple[FrozenSet[Any], float]], min_confidence: float) -> List[Tuple[FrozenSet[Any], FrozenSet[Any], float, float]]:
        '''
            从频繁项集生成关联规则 A → B 并计算支持度和置信度
        '''
        
        support_map: Dict[FrozenSet[Any], float] = {fs: sup for fs, sup in frequent_itemsets}       # 列表转字典
        rules: List[Tuple[FrozenSet[Any], FrozenSet[Any], float, float]] = []
        for fs, sup in frequent_itemsets:
            if len(fs) < 2:             # 单项集没有关联规则
                continue
            subsets = self._proper_subsets(fs)      # 计算fs的真子集
            for a in subsets:
                b = fs.difference(a)        # 计算a的补集, 即规则的右部
                if not b:
                    continue
                conf = sup / support_map.get(a, 1e-12)      # 计算置信度, 并避免除0
                if conf >= min_confidence:
                    rules.append((a, b, sup, conf))         # 仅保留置信度满足阈值的规则(A---conf--->B)
        return rules

    def maximal_itemsets(
        self, frequent_itemsets: List[Tuple[FrozenSet[Any], float]]
    ) -> List[Tuple[FrozenSet[Any], float]]:   # 返回值类型改为携带支持度

        # 同时保留支持度，构建 (fs, support) 的排序列表
        support_map = {fs: sup for fs, sup in frequent_itemsets}
        sets_sorted = sorted(support_map.keys(), key=lambda x: len(x), reverse=True)

        maximal: List[Tuple[FrozenSet[Any], float]] = []

        for i, s in enumerate(sets_sorted):
            if any(s.issubset(t) for j, t in enumerate(sets_sorted) if j < i):
                continue
            maximal.append((s, support_map[s]))  # 返回时带上支持度

        return maximal

    def _all_subsets_frequent(self, candidate: FrozenSet[Any], freq_prev: Dict[FrozenSet[Any], int]) -> bool:
        '''
            检查候选项集 candidate 的所有 (k-1) 子集是否都在上一层频繁项集 freq_prev 中。
            只要有一个子集不频繁，直接返回 False
        '''
        for it in candidate:
            subset = candidate.difference({it})     # 生成 (k-1) 子集, 每次移除一个元素
            if subset not in freq_prev:
                return False
        return True

    def _proper_subsets(self, s: FrozenSet[Any]) -> List[FrozenSet[Any]]:
        '''
            求s的真子集
        '''
        arr = list(s)
        res: List[FrozenSet[Any]] = []
        n = len(arr)
        for mask in range(1, 1 << n):       # [1, 1<<n-1] 枚举所有子集
            if mask == (1 << n) - 1:        # 跳过集合本身(全1)
                continue
            subset = [arr[i] for i in range(n) if (mask >> i) & 1]  # 选择 mask 中为 1 的位对应的元素
            res.append(frozenset(subset))
        return res
