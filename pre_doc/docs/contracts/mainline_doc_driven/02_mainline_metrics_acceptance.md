# Mainline Metrics & Acceptance Contract

## 1. 主线主指标
以下指标构成主线最小验收集合：
- AP / AP_base / AP_novel
- AURC
- SCR
- HPR
- UAR

## 2. 阶段门控
### G0 协议与 baseline
通过条件：
- `Y'(w) ⊂ Y*(w)` 恒成立
- missing-rate 统计与协议配置一致
- closed-world baseline 可运行并输出可复现实验结果

### G1 basis
通过条件：
- Stage B export / bridge / loader 稳定
- class-agnostic track basis 可覆盖主要前景实例

### G2 语义桥
通过条件：
- mixed representation 不弱于单一输入的最小可分性
- Stage C loader / semantic slice 无 schema 漂移

### G3 attribution（核心门控）
通过条件：
- 相对 closed-world baseline，HPR 稳定提升
- 相对 closed-world baseline，UAR 稳定提升
- SCR 不下降，最好有提升
- AP 不出现不可接受的系统性下降

若 G3 未通过，不得进入 refinement 主线。

### G4 linking
通过条件：
- 相邻窗口 linking 稳定
- global 类别汇聚至少不差于 local 平均

### G5 refinement
通过条件：
- AP 或 AP_novel/SCR/HPR/UAR 中至少一组关键指标稳定正增益
- 不出现显著负迁移

## 3. 附录指标
以下指标默认作为补充证据，不作为主线 gate：
- U-OWTA
- class-agnostic HOTA
- STQ-style tracking score

## 4. 自动化验收要求
每个 task pack 必须给出：
- 本阶段主指标
- 最小执行命令
- 产物路径
- PASS / FAIL 判定条件

若没有这些内容，则不视为文档驱动自动化任务。
