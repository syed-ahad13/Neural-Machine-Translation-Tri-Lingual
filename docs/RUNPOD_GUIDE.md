# Neural Machine Translation - RunPod Training Guide

## Quick Start

### 1. Rent RunPod A100
1. Go to [runpod.io](https://runpod.io)
2. Select **A100 80GB** (or A100 40GB)
3. Choose template: **RunPod Pytorch 2.1** or similar
4. Start pod (~$1.99/hr for A100 80GB)

### 2. SSH into Pod
```bash
ssh root@<your-pod-ip> -i ~/.ssh/your_key
```

### 3. Upload Project Files
From your local machine:
```bash
# Compress project
cd /home/onlyahad/Desktop
tar -czvf nmt-project.tar.gz "Neural Machine Translation"

# Upload to RunPod (replace with your pod IP)
scp nmt-project.tar.gz root@<pod-ip>:/workspace/

# On RunPod, extract
ssh root@<pod-ip>
cd /workspace
tar -xzvf nmt-project.tar.gz
mv "Neural Machine Translation" nmt-project
```

### 4. Run Setup
```bash
cd /workspace/nmt-project
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh
```

### 5. Start Training
```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

---

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Model | Mistral-7B-Instruct-v0.3 |
| Method | QLoRA (4-bit) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Batch Size | 4 × 8 = 32 effective |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Max Samples | 100,000 |

## Estimated Time
- **A100 80GB**: ~6-8 hours for 3 epochs
- **A100 40GB**: ~8-10 hours for 3 epochs

## Cost Estimate
- A100 80GB @ $1.99/hr × 8 hrs = **~$16**
- A100 40GB @ $1.49/hr × 10 hrs = **~$15**

---

## After Training

### Download Model
```bash
# On RunPod
cd /workspace/nmt-project
tar -czvf model.tar.gz outputs/

# On local machine
scp root@<pod-ip>:/workspace/nmt-project/model.tar.gz ./
```

### Evaluate Fine-Tuned Model
Use the evaluation notebook with the fine-tuned adapter path.
