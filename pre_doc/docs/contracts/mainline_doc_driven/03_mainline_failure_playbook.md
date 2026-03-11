# Mainline Failure Playbook

本文件定义高风险模块在自动化开发时的标准回退路径。

## 1. Attribution 失败
### 1.1 症状
- unk 吸走过多质量
- bg / unk 分流异常
- HPR / UAR 不升
- coverage 导致 SCR 不升反降

### 1.2 回退顺序
1. 关闭 retrieved candidates
2. 降低 coverage 权重
3. 退化为 row-wise soft assignment + bg/unk + coverage
4. 停止端到端反传，只把 attribution 当 soft pseudo-label 生成器

### 1.3 禁止行为
- 失败时默认继续加新损失项
- 在未过 G3 前直接进入 refinement

## 2. Refinement 失败
### 2.1 症状
- AP 波动大
- HPR 下降
- hidden positives 被误删

### 2.2 回退顺序
1. 去掉 `Q_sem` 对 keep/drop 的主导作用
2. 仅保留 mask / temporal gating
3. 仅做 proposal denoising，不重写语义标签
4. 若仍不稳，整块移出主线，降级到附录

## 3. Linking 失败
### 3.1 症状
- ID switch 过多
- global class 比 local 平均更差

### 3.2 回退顺序
1. 去掉语义项，只保留 IoU + query consistency
2. 取消 memory aggregation
3. 使用 quality-weighted logit average 作为默认 global classification

## 4. Scope 漂移
### 症状
- task pack 未过 gate 却开始实现增强项
- 新模块未说明为何服务于核心命题

### 处理
- 立即停止当前扩展
- 回到 objective/scope contract
- 重新写 task pack，补齐 gate、验收、fallback
