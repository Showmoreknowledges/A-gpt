# 全流程运行指令示例

下列命令串联了前七个阶段：节点属性/LLM 描述 → (可选) 训练 → 推理生成 `rank_lists_final` → 评估与日志。每一步都只依赖 `.npz` 中的内容，无需关心文件名。

## 1. 节点属性计算 & LLM 描述生成
```bash
python scripts/generate_node_descriptions.py \
  --npz-path data/amazon_clothing.npz \
  --output-dir cache/amazon_desc \
  --llm-backend echo \
  --llm-model dummy-llm
```
上述命令会把两个图层的节点属性计算完毕，并调用指定的 LLM（示例中用 echo 后端做快速验证），生成 `cache/amazon_desc/layer_a.jsonl` 与 `layer_b.jsonl`。

## 2. 训练对齐模型（可选，有 InfoNCE/蒸馏）
如果需要进行 InfoNCE/蒸馏训练，可使用仓库自带的训练脚本（参数按需调整）：
```bash
python train_alignment.py \
  --data_path lpformer_data.pt \
  --train_pairs_path train_pairs_merged.pt \
  --test_pairs_path test_pairs_merged.pt \
  --output_dir checkpoints/alignment
```

## 3. 推理：生成 H_a/H_b → 召回 → LLM 重排
```bash
python scripts/run_alignment_inference.py \
  --npz-path data/amazon_clothing.npz \
  --desc-a cache/amazon_desc/layer_a.jsonl \
  --desc-b cache/amazon_desc/layer_b.jsonl \
  --text-model sentence-transformers/all-MiniLM-L6-v2 \
  --save-npz cache/amazon_desc/rank_lists_final.npz \
  --topk 50
```
该命令会：
1. 用 GNN + SentenceTransformer 生成结构/文本表示，经过 Gated Fusion 得到 `H_a/H_b`；
2. 使用 `Retriever` 计算 coarse 候选；
3. 默认调用 echo LLM 进行重排序，输出 `rank_lists_final` 并保存在 `cache/amazon_desc/rank_lists_final.npz`。

## 4. 评估：Hit@k / MRR + 记录 CSV
```bash
python -m src.evaluation.eval_alignment \
  --npz_path data/amazon_clothing.npz \
  --dataset amazon_clothing \
  --rank_path cache/amazon_desc/rank_lists_final.npz \
  --log_csv results/metrics_log.csv
```
执行后终端会打印 Hit@1/5/10 与 MRR，同时 `results/metrics_log.csv` 会自动追加一行（包含日期、时间、dataset、各项指标及 `rank_path` 等信息）。

> 若 `--rank_path` 未提供，评估脚本会使用随机候选做占位，但不会停止运行。