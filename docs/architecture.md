# 系统架构

## 概述

El-Agente-Math 是一个用于检测学术论文中数学公式错误的系统。系统采用 uv workspace 管理的多包 monorepo 架构，由四个核心包组成。

## 包结构

```
packages/
├── agent/          # 公式检测 Agent（Fangshi 开发，冻结）
├── benchmark/      # 评估基准工具
├── data/           # 数据采集与处理
├── meta-agent/     # 元代理：自动迭代改进 Agent
└── openreview-crawler/  # OpenReview 论文爬虫
```

## 核心组件

### 1. Agent (`packages/agent/`)

基于 Pydantic AI 框架的 Agentic Reader，负责判断论文片段是否包含数学公式错误。

- **输入**：MinerU 解析后的 Markdown 论文 + 审稿人标注的可疑片段
- **输出**：`Verdict: FORMULA_ISSUE` 或 `Verdict: NO_FORMULA_ISSUE`
- **工具**：`read_content`、`read_figure`、`search_content`、`update_memo`
- **模板**：Jinja2 系统/用户提示词（`templates/` 目录）

### 2. Benchmark (`packages/benchmark/`)

评估 Agent 检测准确率的完整工具链。

#### 核心功能

| 模块 | 功能 |
|------|------|
| `runner.py` | 批量运行 Agent，支持并发、按论文 ID 过滤、自定义解析目录 |
| `checker.py` | 对比 Agent 输出与人工标注 |
| `generate_negatives.py` | 从 Spotlight 论文采样负例（无公式错误的论文） |
| `confusion_matrix.py` | 计算混淆矩阵（TP/FP/TN/FN/Precision/Recall/F1） |
| `split_dataset.py` | 按论文 ID 将数据集划分为训练集/测试集（防止数据泄露） |

#### 数据模型

- `ConfusionMatrix`：包含 `precision`、`recall`、`f1`、`accuracy` 计算属性
- `SplitManifest`：记录训练/测试集的正例和负例论文 ID

#### 负例生成流程

1. 从 NeurIPS 2025 Spotlight 论文（高质量、无已知问题）中采样
2. 用正则匹配公式块（`$$...$$`、`\begin{equation}` 等）
3. 提取公式上下文，生成与真实 issue 相同格式的文件
4. 78 个负例匹配 78 个正例，构成平衡数据集

### 3. Data (`packages/data/`)

数据采集管线，包括：

- OpenReview 论文爬取
- MinerU API 远程 PDF 解析
- Streamlit UI 人工审核筛选

### 4. Meta-Agent (`packages/meta-agent/`) — 新增

自动化迭代改进 Agent 的元代理系统。

#### 设计动机

基线 Agent 的 F1 仅约 0.53（接近随机猜测），需要系统化的方法来改进检测能力。

#### 架构设计

```
┌─────────────────────────────────────────────────┐
│                  Meta-Agent 循环                 │
│                                                 │
│  1. 分析上一轮失败案例                            │
│     ├── 假阴性（漏检真实错误）                     │
│     └── 假阳性（误报正确公式）                     │
│                                                 │
│  2. 调用 Claude CLI（沙箱化）                     │
│     ├── --cwd workspace/el_agente               │
│     ├── --allowedTools Edit,Read,Write           │
│     └── 提示词含失败案例 + 历史记录                 │
│                                                 │
│  3. 质量检查                                     │
│     └── ruff check                              │
│                                                 │
│  4. 运行基准测试（仅训练集）                       │
│     └── PYTHONPATH=workspace:$PYTHONPATH         │
│                                                 │
│  5. 评估                                        │
│     ├── F1 提升 → 保留改动                        │
│     └── F1 未提升 → 回滚工作区                     │
│                                                 │
│  6. 记录统计 + 生成趋势图                          │
└─────────────────────────────────────────────────┘
```

#### 关键设计决策

1. **工作区隔离**：将 Agent 代码复制到 `output/meta_agent/workspace/el_agente/`，所有修改仅在副本上进行，绝不触碰 `packages/agent/`（Fangshi 的代码）。

2. **PYTHONPATH 遮蔽**：运行基准测试时，通过 `PYTHONPATH=workspace:$PYTHONPATH` 让工作区中的 `el_agente` 模块优先于已安装的版本加载。

3. **数据泄露防护**：
   - Claude CLI 通过 `--cwd workspace/el_agente` 限制文件访问范围
   - 仅允许 `Edit`、`Read`、`Write` 三个工具
   - 编码 Agent 只能看到训练集的失败分析，看不到测试集数据

4. **迭代策略**：每轮只做一个聚焦改动，附带明确假设。保留改进、回滚退步。

#### 模块说明

| 模块 | 职责 |
|------|------|
| `config.py` | `MetaAgentConfig`：路径、预算、可修改文件列表 |
| `history.py` | `IterationRecord` + `MetaHistory`：迭代记录与最佳追踪 |
| `analyzer.py` | 读取 `.result.md` 文件，分类 FN/FP，提取具体失败案例 |
| `orchestrator.py` | 主循环：分析→调用编码 Agent→质检→基准测试→评估 |
| `cli.py` | Typer CLI 入口 |
| `templates/meta_iterate.jinja2` | 编码 Agent 的提示词模板 |

#### 输出

- `output/meta_agent/stats.csv`：每轮指标记录
- `output/meta_agent/trend.txt`：ASCII 趋势图
- `output/meta_agent/trend.png`：Matplotlib 趋势图
- `output/meta_agent/history.json`：完整迭代历史
- `output/meta_agent/best_agent/`：最优 Agent 代码副本

## 基线评估结果

在 78 正例 + 78 负例的完整数据集上：

| 指标 | 值 |
|------|-----|
| Precision | 0.5571 |
| Recall | 0.5065 |
| F1 | 0.5306 |
| Accuracy | 0.5548 |

训练/测试划分：按论文 ID 50/50 分割（seed=42），正例 23/24 篇论文，负例 39/39 篇。

## 技术栈

- **包管理**：uv workspace
- **LLM 框架**：Pydantic AI
- **PDF 解析**：MinerU API
- **数据模型**：Pydantic v2
- **模板**：Jinja2
- **CLI**：Typer
- **代码质量**：ruff + mypy
- **编码 Agent**：Claude CLI (`claude --print`)
