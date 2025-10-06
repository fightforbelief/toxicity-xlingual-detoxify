# 评分文档 (Scoring Documentation)

## 评估指标 (Evaluation Metrics)

本项目使用多种指标来评估多语言毒性检测模型的性能，确保全面和公平的比较。

## 主要指标 (Primary Metrics)

### 1. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **定义**: 真正率(TPR)与假正率(FPR)曲线下的面积
- **范围**: 0.0 - 1.0 (1.0为完美分类)
- **解释**: 
  - 0.5: 随机分类器
  - 0.7-0.8: 良好性能
  - 0.8-0.9: 优秀性能
  - >0.9: 卓越性能

### 2. F1-Score
- **定义**: 精确率和召回率的调和平均数
- **公式**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **范围**: 0.0 - 1.0
- **优势**: 平衡精确率和召回率，适合不平衡数据集

### 3. 精确率 (Precision)
- **定义**: 预测为毒性中实际为毒性的比例
- **公式**: Precision = TP / (TP + FP)
- **重要性**: 减少误报，避免将正常内容标记为毒性

### 4. 召回率 (Recall)
- **定义**: 实际毒性中被正确识别的比例
- **公式**: Recall = TP / (TP + FN)
- **重要性**: 减少漏报，确保捕获更多毒性内容

## 辅助指标 (Secondary Metrics)

### 5. 准确率 (Accuracy)
- **定义**: 正确预测的总比例
- **公式**: Accuracy = (TP + TN) / (TP + TN + FP + FN)
- **注意**: 在不平衡数据集中可能误导

### 6. 平衡准确率 (Balanced Accuracy)
- **定义**: 敏感性和特异性的平均值
- **公式**: Balanced Accuracy = (Sensitivity + Specificity) / 2
- **优势**: 对类别不平衡更鲁棒

### 7. 平均精确率 (Average Precision)
- **定义**: 精确率-召回率曲线下的面积
- **范围**: 0.0 - 1.0
- **优势**: 对类别不平衡数据集更敏感

## 多语言评估 (Multilingual Evaluation)

### 语言特定评估
每种语言独立计算所有指标：
```bash
# 评估特定语言
python code/score.py --config configs/detoxify_multilingual.yaml \
                     --pred_dir output/predictions/baseline \
                     --languages en es fr de it pt ru tr \
                     --output output/evaluation_results.csv
```

### 宏平均 (Macro Average)
计算所有语言指标的平均值：
- **宏平均F1**: 各语言F1分数的平均值
- **宏平均ROC-AUC**: 各语言ROC-AUC的平均值
- **优势**: 给予每种语言相等权重

### 微平均 (Micro Average)
基于所有语言的合并预测计算指标：
- **微平均F1**: 基于所有预测的F1分数
- **微平均ROC-AUC**: 基于所有预测的ROC-AUC
- **优势**: 反映整体性能

## 阈值优化 (Threshold Optimization)

### 默认阈值
- **标准阈值**: 0.5
- **适用场景**: 平衡精确率和召回率

### 优化策略
```python
# 基于F1分数优化阈值
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

### 语言特定阈值
每种语言独立优化阈值：
- 考虑语言特定的数据分布
- 适应不同语言的毒性模式
- 提高整体性能

## 评估脚本使用 (Evaluation Script Usage)

### 基本用法
```bash
# 评估单个模型
python code/score.py \
    --config configs/detoxify_multilingual.yaml \
    --pred_dir output/predictions/baseline \
    --languages en es fr de it pt ru tr \
    --output output/baseline_evaluation.csv
```

### 比较多个模型
```bash
# 评估多个模型
python code/score.py \
    --config configs/detoxify_multilingual.yaml \
    --pred_dir output/predictions \
    --models baseline detoxify_original detoxify_translated \
    --languages en es fr de it pt ru tr \
    --output output/model_comparison.csv
```

### 高级选项
```bash
# 自定义评估
python code/score.py \
    --config configs/detoxify_multilingual.yaml \
    --pred_dir output/predictions \
    --languages en es fr de it pt ru tr \
    --models baseline detoxify_original \
    --output output/custom_evaluation.csv \
    --validate_predictions \
    --create_plots
```

## 预测文件格式 (Prediction File Format)

### 标准格式
```csv
prediction,label
0.85,1
0.23,0
0.91,1
0.12,0
```

- `prediction`: 模型预测的概率分数 (0.0 - 1.0)
- `label`: 真实标签 (0 或 1)

### 文件命名约定
```
output/predictions/
├── baseline/
│   ├── en_predictions.csv
│   ├── es_predictions.csv
│   └── ...
├── detoxify_original/
│   ├── en_predictions.csv
│   ├── es_predictions.csv
│   └── ...
└── detoxify_translated/
    ├── en_predictions.csv
    ├── es_predictions.csv
    └── ...
```

## 结果解释 (Result Interpretation)

### 性能等级
| ROC-AUC | F1-Score | 性能等级 | 描述 |
|---------|----------|----------|------|
| 0.5-0.6 | 0.3-0.5  | 差       | 略好于随机 |
| 0.6-0.7 | 0.5-0.6  | 一般     | 基本可用 |
| 0.7-0.8 | 0.6-0.7  | 良好     | 实用水平 |
| 0.8-0.9 | 0.7-0.8  | 优秀     | 高质量 |
| >0.9    | >0.8     | 卓越     | 接近完美 |

### 跨语言性能分析
1. **语言难度**: 不同语言的性能差异
2. **数据质量**: 训练数据对性能的影响
3. **模型适应性**: 模型对不同语言的适应性
4. **翻译影响**: 翻译数据对性能的影响

## 统计显著性测试 (Statistical Significance Testing)

### McNemar测试
比较两个模型的分类结果：
```python
from statsmodels.stats.contingency_tables import mcnemar

# 比较模型A和模型B
statistic, p_value = mcnemar(confusion_matrix, exact=True)
if p_value < 0.05:
    print("性能差异显著")
```

### 置信区间
计算指标的95%置信区间：
```python
from scipy import stats

# 计算ROC-AUC的置信区间
ci = stats.bootstrap((y_true, y_pred), roc_auc_score, 
                    n_resamples=1000, confidence_level=0.95)
```

## 可视化 (Visualization)

### ROC曲线
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### 混淆矩阵
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

## 最佳实践 (Best Practices)

1. **多指标评估**: 不要仅依赖单一指标
2. **跨语言验证**: 确保模型在所有语言上表现良好
3. **阈值优化**: 根据应用场景优化分类阈值
4. **统计测试**: 使用统计测试验证性能差异
5. **可视化分析**: 使用图表深入理解模型行为
6. **错误分析**: 分析错误案例以改进模型

## 常见问题 (FAQ)

### Q: 为什么F1分数比准确率更重要？
A: 在不平衡数据集中，F1分数更好地反映了模型在少数类（毒性内容）上的性能。

### Q: 如何处理不同语言的性能差异？
A: 使用宏平均确保每种语言得到相等权重，同时分析语言特定的性能模式。

### Q: 什么时候使用微平均？
A: 当您关心整体性能而不是语言特定性能时，微平均更合适。

### Q: 如何解释ROC-AUC为0.5的结果？
A: ROC-AUC为0.5表示模型性能与随机分类器相同，需要重新训练或调整模型。
