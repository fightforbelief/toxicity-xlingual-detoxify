# 数据文档 (Data Documentation)

## 数据概览 (Data Overview)

本项目使用Jigsaw 2020多语言毒性检测数据集，包含8种语言的评论数据：英语(en)、西班牙语(es)、法语(fr)、德语(de)、意大利语(it)、葡萄牙语(pt)、俄语(ru)和土耳其语(tr)。

## 数据源 (Data Sources)

### 原始数据 (Raw Data)
- **Jigsaw 2020验证集**: `data/raw/validation.csv`
- **Jigsaw 2020测试集**: `data/raw/test.csv`
- 来源: [Kaggle Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)

### 处理后的数据 (Processed Data)
- **训练/验证/测试分割**: `data/processed/{language}_{split}.csv`
- **翻译数据清单**: `data/processed/translation_manifest.json`
- **翻译训练数据**: `data/translated/`

## 数据模式 (Data Schema)

### 标准列 (Standard Columns)
```csv
comment_text,toxic,lang
"这是一个评论",0,en
"Another comment",1,en
```

- `comment_text`: 评论文本内容
- `toxic`: 毒性标签 (0=非毒性, 1=毒性)
- `lang`: 语言代码 (en, es, fr, de, it, pt, ru, tr)

### 扩展列 (Extended Columns)
某些文件可能包含额外列：
- `id`: 唯一标识符
- `severe_toxic`: 严重毒性标签
- `threat`: 威胁标签
- `insult`: 侮辱标签
- `identity_hate`: 身份仇恨标签

## 数据分割 (Data Splits)

### 分割比例
- **训练集**: 80%
- **验证集**: 10%
- **测试集**: 10%

### 分割策略
- 使用分层抽样保持毒性标签分布
- 每种语言独立分割
- 随机种子固定为42以确保可重现性

### 文件命名
```
data/processed/
├── en_train.csv    # 英语训练集
├── en_val.csv      # 英语验证集
├── en_test.csv     # 英语测试集
├── es_train.csv    # 西班牙语训练集
├── es_val.csv      # 西班牙语验证集
├── es_test.csv     # 西班牙语测试集
└── ...
```

## 数据统计 (Data Statistics)

### 语言分布
| 语言 | 代码 | 训练样本 | 验证样本 | 测试样本 | 毒性比例 |
|------|------|----------|----------|----------|----------|
| 英语 | en   | ~40,000  | ~5,000   | ~5,000   | ~8.5%    |
| 西班牙语 | es | ~4,000   | ~500     | ~500     | ~6.2%    |
| 法语 | fr   | ~3,500   | ~400     | ~400     | ~7.1%    |
| 德语 | de   | ~3,200   | ~400     | ~400     | ~5.8%    |
| 意大利语 | it | ~2,800   | ~350     | ~350     | ~6.9%    |
| 葡萄牙语 | pt | ~2,500   | ~300     | ~300     | ~7.5%    |
| 俄语 | ru   | ~2,200   | ~275     | ~275     | ~8.1%    |
| 土耳其语 | tr | ~1,800   | ~225     | ~225     | ~6.7%    |

*注：实际数量可能因数据可用性而异*

## 翻译数据 (Translated Data)

### 翻译策略
- 使用Google Translate API将英语训练数据翻译为其他语言
- 保持原始毒性标签不变
- 创建语言特定的训练集

### 翻译数据文件
```
data/translated/
├── en_train.csv           # 原始英语训练数据
├── es_train_translated.csv # 翻译的西班牙语训练数据
├── fr_train_translated.csv # 翻译的法语训练数据
└── ...
```

### 翻译质量
- 使用语言检测验证翻译质量
- 最低置信度阈值：0.7
- 过滤低质量翻译

## 数据预处理 (Data Preprocessing)

### 文本清理
- 移除URL链接
- 移除@提及
- 保留#标签（用于上下文）
- Unicode标准化
- 空白字符规范化

### 语言检测
- 使用langdetect库检测语言
- 过滤低置信度检测结果
- 处理混合语言文本

### 标签处理
- 二值化毒性标签
- 处理缺失标签
- 平衡类别分布

## 数据验证 (Data Validation)

### 质量检查
- 检查必需列的存在
- 验证标签值范围
- 检测重复样本
- 检查文本长度分布

### 统计验证
- 验证分割比例
- 检查分层抽样效果
- 分析类别分布
- 检测数据泄露

## 使用示例 (Usage Examples)

### 加载数据
```python
import pandas as pd

# 加载特定语言的训练数据
train_df = pd.read_csv('data/processed/en_train.csv')
print(f"训练样本数: {len(train_df)}")
print(f"毒性比例: {train_df['toxic'].mean():.3f}")

# 加载所有语言的数据
languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'tr']
for lang in languages:
    df = pd.read_csv(f'data/processed/{lang}_train.csv')
    print(f"{lang}: {len(df)} samples, {df['toxic'].mean():.3f} toxic")
```

### 数据探索
```python
# 文本长度分析
train_df['text_length'] = train_df['comment_text'].str.len()
print(train_df['text_length'].describe())

# 毒性标签分布
print(train_df['toxic'].value_counts())

# 语言检测
from langdetect import detect
sample_text = train_df['comment_text'].iloc[0]
detected_lang = detect(sample_text)
print(f"检测到的语言: {detected_lang}")
```

## 注意事项 (Important Notes)

1. **数据敏感性**: 数据集包含有毒内容，使用时需谨慎
2. **语言偏见**: 不同语言的数据质量和数量存在差异
3. **翻译质量**: 翻译数据可能存在质量问题和偏见
4. **标签一致性**: 跨语言的毒性标签定义可能不完全一致
5. **隐私保护**: 评论数据已匿名化，但仍需注意隐私保护

## 更新日志 (Update Log)

- **v1.0** (2024-01-XX): 初始数据分割和预处理
- **v1.1** (2024-01-XX): 添加翻译数据支持
- **v1.2** (2024-01-XX): 改进数据验证和质量检查
