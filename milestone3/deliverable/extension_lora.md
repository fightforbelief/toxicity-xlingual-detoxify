````markdown
Milestone 3 Extension: LoRA Fine-Tuning for Multilingual Toxicity Detection

This document explains how to use the **LoRA-based fine-tuning extension** for our multilingual toxicity detection project.

The extension adds a **parameter-efficient fine-tuning (LoRA)** layer on top of the existing `xlm-roberta-base` classifier, and trains **separate adapters per language** using **class-weighted BCEWithLogitsLoss** to handle label imbalance.  
This replaces full-model training with a lighter, cheaper, and faster alternative while maintaining strong performance.

---

## Quick Start (TL;DR)

**Train LoRA adapters for all languages:**
```bash
python code/lora_finetune.py --config configs/lora_finetune.yaml
````

**Train only Spanish (example):**

```bash
python code/lora_finetune.py --config configs/lora_finetune.yaml --languages es
```

**Single-text inference:**

```bash
python code/lora_inference.py --config configs/lora_finetune.yaml --lang es --text "hola idiota"
```

**CSV inference:**

```bash
python code/lora_inference.py \
  --config configs/lora_finetune.yaml \
  --lang es \
  --input_csv data/processed/es/test.csv \
  --output_csv output/predictions/lora_es_test.csv
```

---

## 1. File Overview

### **`code/lora_finetune.py`**

LoRA-based fine-tuning script:

* Loads per-language datasets from `data/processed/{lang}/`
* Tokenizes text using XLM-R tokenizer
* Wraps `xlm-roberta-base` with LoRA adapters (via `peft`)
* Computes **positive-class weight** and trains with weighted BCE
* Saves trained LoRA adapters to:

  ```
  output/runs/lora/{lang}/lora_adapter/
  ```

### **`code/lora_inference.py`**

Inference script:

* Loads base model + LoRA adapter
* Runs prediction on:

  * a single input text
  * a full CSV file
* Outputs toxicity probability and binary label

### **`configs/lora_finetune.yaml`**

Configuration for:

* Data directory and language list
* Model checkpoint (`xlm-roberta-base`)
* Max token length
* Training parameters (batch size, learning rate, epochs)
* Output directory

---

## 2. Expected Directory Structure

```
.
├── code/
│   ├── lora_finetune.py
│   └── lora_inference.py
├── configs/
│   └── lora_finetune.yaml
├── data/
│   └── processed/
│       ├── es/
│       │   ├── train.csv
│       │   ├── val.csv
│       │   └── test.csv
│       ├── it/
│       │   ├── train.csv
│       │   ├── val.csv
│       │   └── test.csv
│       └── tr/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
└── output/
    └── runs/
        └── lora/
            └── {lang}/
                └── lora_adapter/
```

Each `train.csv` and `val.csv` must contain:

* `comment_text`
* `toxic` (0/1 labels)

---

## 3. Environment Setup

Install dependencies:

```bash
pip install torch transformers datasets peft pyyaml pandas
```

Notes:

* Apple Silicon will use `mps` automatically.
* On Colab, set runtime → GPU.

---

## 4. Configuration Details

The YAML controls all key settings:

```yaml
data:
  processed_dir: "data/processed"
  languages: ["es", "it", "tr"]
  text_col: "comment_text"
  label_col: "toxic"

model:
  base_model_name: "xlm-roberta-base"
  max_length: 512

training:
  output_dir: "output/runs/lora"
  batch_size: 16
  lr: 2e-5
  epochs: 3
  weight_decay: 0.01
```

You can override the language list at runtime:

```bash
--languages es it tr
```

---

## 5. Running LoRA Fine-Tuning

### **Train all languages**

```bash
python code/lora_finetune.py --config configs/lora_finetune.yaml
```

### **Train a subset**

```bash
python code/lora_finetune.py --config configs/lora_finetune.yaml --languages es it
```

### **Specify device**

```bash
python code/lora_finetune.py --config configs/lora_finetune.yaml --device cuda
```

During training, you will see:

* Class imbalance weight
* Trainable parameters (only LoRA layers)
* Evaluation scores at checkpoints

Outputs will be saved here:

```
output/runs/lora/{lang}/lora_adapter/
```

---

## 6. Running Inference

### **Single text**

```bash
python code/lora_inference.py \
  --config configs/lora_finetune.yaml \
  --lang es \
  --text "Eres un idiota."
```

Output:

```
Toxicity prob: 0.8732, label@0.5: 1
```

### **CSV file**

```bash
python code/lora_inference.py \
  --config configs/lora_finetune.yaml \
  --lang it \
  --input_csv data/processed/it/test.csv \
  --output_csv output/predictions/lora_it_test.csv
```

Adds:

* `toxicity_prob`
* `toxicity_label` (threshold=0.5 by default)

---

## 7. What This Extension Adds (Summary)

This extension implements the **LoRA (Low-Rank Adaptation)** method as a lightweight alternative to full fine-tuning. Instead of updating all ~270M parameters of XLM-R, LoRA injects small trainable matrices into attention layers, reducing trainable parameters by >99%.

Advantages:

* Faster training
* Can train each language separately
* Lower GPU/memory requirements
* Works well with small data
* Keeps the base model shared across languages

Additionally, class imbalance is addressed by computing **positive class weights** from the training set and using weighted BCEWithLogitsLoss to up-weight toxic examples.

This improves recall on minority toxic samples.

---

## 8. Troubleshooting

**Tokenizer mismatch error?**
Delete the adapter folder and re-train.

**CUDA out of memory?**
Reduce batch size inside YAML.

**Device issues on Mac?**
Run:

```bash
--device mps
```

---

## 9. Credits

This extension builds on:

* HuggingFace Transformers
* PEFT LoRA library
* M2 XLM-R baseline code

```
