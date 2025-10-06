# 多语言毒性检测项目 (Multilingual Toxicity Detection)

基于Detoxify的多语言毒性检测项目，支持8种语言的毒性内容识别和分类。

## 项目概述 (Project Overview)

本项目旨在构建一个强大的多语言毒性检测系统，能够准确识别英语、西班牙语、法语、德语、意大利语、葡萄牙语、俄语和土耳其语中的毒性内容。项目采用两种主要方法：

1. **原始语言方法**: 使用每种语言的原始数据进行训练
2. **翻译方法**: 将英语数据翻译为其他语言进行训练

## 项目结构 (Project Structure)

```
toxicity-xlingual-detoxify/
├── code/                           # 核心代码
│   ├── simple_baseline.py         # TF-IDF + 逻辑回归基线模型
│   ├── score.py                   # 统一评估脚本
│   └── scripts/                   # 数据处理脚本
│       ├── make_multilingual_splits.py    # 创建多语言数据分割
│       └── prepare_translated_index.py    # 准备翻译数据清单
├── configs/                        # 配置文件
│   ├── detoxify_multilingual.yaml # 基础配置
│   ├── translated.yaml            # 翻译方法配置
│   └── original.yaml              # 原始语言方法配置
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据 (Git忽略)
│   └── processed/                 # 处理后的数据分割
├── external/                      # 外部依赖
│   └── detoxify/                  # Detoxify子模块
├── output/                        # 输出目录
│   ├── runs/                      # 模型检查点和日志
│   └── predictions/               # 预测结果
├── docs/                          # 文档
│   ├── data.md                    # 数据文档
│   └── scoring.md                 # 评估指标文档
├── .gitignore                     # Git忽略文件
├── requirements.txt               # Python依赖
└── README.md                      # 项目说明
```

## 快速开始 (Quick Start)

### 1. 环境设置 (Environment Setup)

```bash
# 克隆项目
git clone <repository-url>
cd toxicity-xlingual-detoxify

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备 (Data Preparation)

```bash
# 下载Jigsaw 2020数据到data/raw/目录
# 然后创建多语言数据分割
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en es fr de it pt ru tr
```

### 3. 训练基线模型 (Train Baseline Model)

```bash
# 训练TF-IDF + 逻辑回归基线模型
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en es fr de it pt ru tr
```

### 4. 评估模型 (Evaluate Models)

```bash
# 评估模型性能
python code/score.py \
    --config configs/detoxify_multilingual.yaml \
    --pred_dir output/predictions \
    --languages en es fr de it pt ru tr \
    --output output/evaluation_results.csv
```

## 支持的语言 (Supported Languages)

| 语言 | 代码 | 状态 | 说明 |
|------|------|------|------|
| 英语 | en | ✅ | 主要语言，数据最丰富 |
| 西班牙语 | es | ✅ | 支持完整 |
| 法语 | fr | ✅ | 支持完整 |
| 德语 | de | ✅ | 支持完整 |
| 意大利语 | it | ✅ | 支持完整 |
| 葡萄牙语 | pt | ✅ | 支持完整 |
| 俄语 | ru | ✅ | 支持完整 |
| 土耳其语 | tr | ✅ | 支持完整 |

## 模型架构 (Model Architecture)

### 基线模型 (Baseline Model)
- **TF-IDF向量化**: 提取文本特征
- **逻辑回归**: 二分类器
- **语言特定训练**: 每种语言独立训练

### Detoxify模型 (Detoxify Model)
- **预训练BERT**: 基于Transformer的编码器
- **多语言支持**: 支持跨语言理解
- **微调策略**: 针对毒性检测任务优化

## 评估指标 (Evaluation Metrics)

- **ROC-AUC**: 主要评估指标
- **F1-Score**: 平衡精确率和召回率
- **精确率**: 减少误报
- **召回率**: 减少漏报
- **宏平均**: 跨语言平均性能
- **微平均**: 整体性能

## 使用方法 (Usage)

### 训练模型 (Training)

```bash
# 原始语言方法
python train.py --config configs/original.yaml

# 翻译方法
python train.py --config configs/translated.yaml
```

### 预测 (Prediction)

```bash
# 单语言预测
python predict.py --model output/runs/baseline_en --text "Your text here"

# 多语言预测
python predict.py --model output/runs/detoxify_multilingual --text "Your text here" --language en
```

### 评估 (Evaluation)

```bash
# 全面评估
python code/score.py --config configs/detoxify_multilingual.yaml --pred_dir output/predictions

# 特定语言评估
python code/score.py --config configs/detoxify_multilingual.yaml --languages en es fr
```

## 配置说明 (Configuration)

### 基础配置 (Base Configuration)
- `configs/detoxify_multilingual.yaml`: 包含所有基础设置
- 数据路径、模型参数、训练设置等

### 方法特定配置 (Method-Specific Configuration)
- `configs/original.yaml`: 原始语言方法配置
- `configs/translated.yaml`: 翻译方法配置

### 自定义配置 (Custom Configuration)
```yaml
# 示例：自定义配置
data:
  languages: ["en", "es", "fr"]  # 只使用特定语言
  
model:
  batch_size: 32                # 调整批次大小
  learning_rate: 1e-5           # 调整学习率
  
training:
  num_epochs: 5                 # 调整训练轮数
```

## 数据格式 (Data Format)

### 输入格式 (Input Format)
```csv
comment_text,toxic,lang
"This is a comment",0,en
"Another comment",1,en
```

### 预测格式 (Prediction Format)
```csv
prediction,label
0.85,1
0.23,0
```

## 性能基准 (Performance Benchmarks)

### 基线模型性能 (Baseline Model Performance)
| 语言 | ROC-AUC | F1-Score | 精确率 | 召回率 |
|------|---------|----------|--------|--------|
| 英语 | 0.85 | 0.72 | 0.68 | 0.76 |
| 西班牙语 | 0.82 | 0.69 | 0.65 | 0.73 |
| 法语 | 0.81 | 0.67 | 0.63 | 0.71 |
| 德语 | 0.83 | 0.70 | 0.66 | 0.74 |
| 意大利语 | 0.80 | 0.66 | 0.62 | 0.70 |
| 葡萄牙语 | 0.79 | 0.65 | 0.61 | 0.69 |
| 俄语 | 0.78 | 0.64 | 0.60 | 0.68 |
| 土耳其语 | 0.77 | 0.63 | 0.59 | 0.67 |

*注：性能数据为示例，实际结果可能因数据质量和模型配置而异*

## 贡献指南 (Contributing)

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证 (License)

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢 (Acknowledgments)

- [Detoxify](https://github.com/unitaryai/detoxify) - 基础毒性检测模型
- [Jigsaw](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) - 多语言毒性检测数据集
- [Hugging Face Transformers](https://huggingface.co/transformers/) - 预训练模型库

## 联系方式 (Contact)

如有问题或建议，请通过以下方式联系：

- 创建 [Issue](https://github.com/your-repo/issues)
- 发送邮件至: your-email@example.com

## 更新日志 (Changelog)

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 支持8种语言的毒性检测
- 实现基线模型和Detoxify模型
- 完整的评估和可视化工具

### v0.1.0 (2024-01-XX)
- 项目初始化
- 基础架构搭建
- 数据预处理管道