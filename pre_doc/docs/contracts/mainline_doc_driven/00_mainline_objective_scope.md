# Mainline Objective & Scope Contract

## 1. 核心命题
主线唯一要证明的命题是：

> 在仅有不完备 window-level positive label set 的 WS-OVVIS 设定下，必须做 open-world set-to-track attribution；若继续采用 closed-world 弱监督，hidden positives 会系统性塌到背景。

## 2. 主文必须保留的核心工作
主文默认必须聚焦以下四块：
1. WS-OVVIS 任务设定与缺失协议；
2. class-agnostic video instance basis；
3. mixed track representation + open-world attribution；
4. hidden-positive / unknown handling 的证据（SCR/HPR/UAR）。

## 3. 主线默认保留范围
- Uniform Missing / Long-tail Missing
- Stage B export / bridge / loader
- Stage C loader / semantic slice / attribution mainline
- adjacent-window linking
- single-round refinement（只有在 G4 通过后进入主线）

## 4. 默认关闭范围
以下项默认不进入主线：
- Scenario/Domain Missing 主协议
- label-set expansion
- 第二轮 refinement
- 强 memory-based global classification
- LLM 候选重排
- 大范围 joint fine-tune

## 5. 什么时候允许扩大 scope
只有满足以下任一条件时，增强项才允许进入：
- 已通过对应 gate，且核心命题已经被稳定支撑；
- 增强项是附录/补充实验，不会污染主线默认实现；
- 指标与失败回退策略已在 task pack 中显式写明。

## 6. 不允许 Codex 自主拍板的事项
- 协议定义与数据切窗规则
- 主文指标口径
- 主线默认 ON/OFF 模块
- 是否进入下一阶段 gate
- 是否把增强项提升为主线实现
