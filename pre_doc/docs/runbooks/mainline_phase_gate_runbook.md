# Mainline Phase/Gate Runbook

## 1. 作用
本 runbook 把主线开发拆成固定阶段，并规定每一阶段的目标、输入、产物、验收和下一阶段进入条件。

## 2. 阶段定义
### Phase 0 / G0：协议与基线
输入：现有协议工具、baseline 训练脚本
输出：`Y'(w)` cache、missing 统计、closed-world baseline、pseudo tube + CLIP baseline
验收：见 metrics contract

### Phase 1 / G1：class-agnostic basis
输入：伪 tube、Stage B export 合同与工具
输出：basis checkpoint、track feature export、bridge input、基础诊断
验收：bridge / loader 稳定，track basis 质量达标

### Phase 2 / G2：语义桥
输入：Stage B 导出产物、文本原型缓存、Stage C loader
输出：mixed representation、semantic slice、最小语义可分性报告
验收：混合表征与 loader 稳定

### Phase 3 / G3：attribution 核心验证
输入：G2 产物
输出：closed-world vs open-world vs +coverage 对比、SCR/HPR/UAR/AURC
验收：核心命题成立

### Phase 4 / G4：整视频闭环
输入：G3 通过的 attribution 主线
输出：adjacent-window linking、global track、full-video 输出
验收：linking 稳定，global classification 不退化

### Phase 5 / G5：单轮 refinement
输入：G4 稳定主线
输出：single-round refinement 结果与收益分析
验收：有净收益且无明显负迁移

## 3. 每阶段 task pack 必填项
- 当前 phase / gate
- 输入依赖
- 可修改文件范围
- 默认关闭项
- 验收命令
- 失败 fallback 顺序
- 产物路径

## 4. 任务推进纪律
- 不允许跳 gate
- 不允许在未通过 G3 时默认进入 refinement
- 不允许在未通过 G4 时默认启用 memory aggregation
- 失败后先回退，不先扩 scope
