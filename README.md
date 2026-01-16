```
⚡ master ~ python train_distill.py \
  --train-jsonl ./train.jsonl \
  --per-device-train-batch-size 32 \
  --logging-steps 10 > output.log 2>&1
```