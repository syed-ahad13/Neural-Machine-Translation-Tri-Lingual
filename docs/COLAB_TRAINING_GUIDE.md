# Colab Training - Step-by-Step Guide

## Prerequisites
- Google account
- Google Colab (free) or Colab Pro+ (recommended for A100)
- Hugging Face account with token

---

## Step 1: Upload Training Data to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Upload your `train.jsonl` file (532MB)
   - Location: `/home/onlyahad/Desktop/Neural Machine Translation/data/train.jsonl`
3. Note where you uploaded it (root or a folder)

---

## Step 2: Open the Training Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook
3. Select: `notebooks/02_finetune_llama_factory.ipynb`

---

## Step 3: Configure GPU

1. Runtime → Change runtime type
2. Select: **T4 GPU** (free) or **A100** (Pro+)
3. Click Save

---

## Step 4: Run the Notebook

Run cells in order:

| Cell | What it does | Time |
|------|--------------|------|
| 1 | Install LLaMA-Factory | ~3 min |
| 2 | Check GPU | ~5 sec |
| 3 | Mount Google Drive | ~10 sec |
| 4 | Copy training data | ~1 min |
| 5 | Create dataset config | ~5 sec |
| 6 | Login to HuggingFace | ~30 sec |
| 7 | Create training config | ~5 sec |
| 8 | **START TRAINING** | **8-60 hours** |

---

## Step 5: Handle Session Disconnects

Colab sessions can disconnect. If this happens:

1. Re-open the notebook
2. Run cells 1-7 again (they're fast)
3. Run cell 8 (training) - it will **resume from last checkpoint**

Checkpoints are saved to Google Drive every 500 steps.

---

## Step 6: After Training

1. Run cell 9 to export the model
2. Run cells 10-11 to test translations
3. Download from Google Drive:
   - `nmt-outputs/mistral-nmt-qlora/` (LoRA adapter)
   - `nmt-outputs/mistral-nmt-merged/` (full model)

---

## Estimated Time & Cost

| GPU | Time | Cost |
|-----|------|------|
| T4 (free) | 40-60 hours | Free (multiple sessions) |
| A100 (Pro+) | 8-10 hours | $50/month subscription |

---

## Troubleshooting

**"CUDA out of memory"**
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_samples` to 50000

**"Session disconnected"**
- Normal! Just reconnect and re-run. Checkpoints are saved.

**"Model not found"**
- Make sure you ran the HuggingFace login cell
- Accept Mistral's terms at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
