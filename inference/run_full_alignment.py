# src/inference/run_full_alignment.py

from __future__ import annotations
import json
import torch
import numpy as np
from typing import List, Dict

from tqdm import tqdm  # ★ 新增：进度条

# ------------------------------
#  Import from your own modules
# ------------------------------
from src.data.load_npz import load_graph_pair
from src.models.gnn_encoder import GNNEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import GatedFusion
from src.models.retrieval import Retriever
from src.llm.reranker import LLMReranker
from src.llm.providers import OpenAIClient


# ============================================================
#                 1. 加载节点描述（LLM 生成）
# ============================================================

def load_descriptions(desc_path: str) -> Dict[int, Dict[str, str]]:
    """
    desc_path: e.g., data/prompts/douban_layerA.jsonl
    返回: {id: {"struct_desc": ..., "sem_desc": ...}}
    """
    desc_map: Dict[int, Dict[str, str]] = {}
    with open(desc_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            node_id = int(obj["id"])
            desc_map[node_id] = {
                "struct_desc": obj.get("struct_desc", ""),
                "sem_desc": obj.get("sem_desc", "")
            }
    return desc_map


# ============================================================
#   2. 主函数：加载模型 → 得到 H_a,H_b → 召回 → LLM 重排
# ============================================================

def run_full_alignment(
    npz_path: str,
    desc_a_path: str,
    desc_b_path: str,
    topk: int = 50,
    device: str = "cuda",
    llm_model_name: str = "gpt-5.1-mini",
    embedding_text_model: str = "all-MiniLM-L6-v2",
    show_progress: bool = True,
):
    """
    完整流程（推理阶段）:
    1. 加载 npz 数据集 (GraphPair)
    2. 加载节点 LLM 描述 (struct_desc / sem_desc)
    3. 结构编码 (GNN)
    4. 文本编码 (SentenceTransformer)
    5. 多模态融合 → H_a, H_b
    6. 多信号召回 → coarse top-K candidates
    7. LLM pairwise rerank → final rank_lists

    参数:
        npz_path:        任意满足约定格式的 .npz 路径（不局限于 douban）
        desc_a_path:     Layer A 的描述 jsonl
        desc_b_path:     Layer B 的描述 jsonl
        topk:            每个源节点保留多少候选
        device:          "cuda" or "cpu"
        llm_model_name:  用于重排序的 LLM 名字
        embedding_text_model: 文本编码模型（SentenceTransformer 名字）
        show_progress:   是否显示 tqdm 进度条

    返回:
        rank_lists_final: np.ndarray, shape = [num_nodes_layerA, topk]
    """

    # --------------------------------------------------------
    # (0) 加载数据
    # --------------------------------------------------------
    print(f"[1/7] Loading dataset from {npz_path} ...")
    graph_pair = load_graph_pair(npz_path)
    g1, g2 = graph_pair.g1, graph_pair.g2
    num_a, num_b = g1.num_nodes, g2.num_nodes
    print(f"       Layer A nodes: {num_a}, Layer B nodes: {num_b}")

    # --------------------------------------------------------
    # (1) 加载节点描述（LLM 的结构/语义描述）
    # --------------------------------------------------------
    print(f"[2/7] Loading descriptions ...")
    print(f"       Layer A desc: {desc_a_path}")
    print(f"       Layer B desc: {desc_b_path}")
    desc_a = load_descriptions(desc_a_path)
    desc_b = load_descriptions(desc_b_path)

    assert len(desc_a) == num_a, f"desc_a 数量 {len(desc_a)} 与 g1.num_nodes {num_a} 不符"
    assert len(desc_b) == num_b, f"desc_b 数量 {len(desc_b)} 与 g2.num_nodes {num_b} 不符"

    # --------------------------------------------------------
    # (2) 文本编码模块（SentenceTransformer）
    # --------------------------------------------------------
    print(f"[3/7] Encoding text with SentenceTransformer: {embedding_text_model} ...")
    text_encoder = TextEncoder(model_name=embedding_text_model)

    def encode_descriptions(desc_map: Dict[int, Dict[str, str]],
                        text_encoder) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        返回:
        H_text_struct: [N, d_text]
        H_text_sem:    [N, d_text] 或 None （当数据集中没有语义描述时）
        """
        ids = sorted(desc_map.keys())
        struct_list = [desc_map[i].get("struct_desc", "") for i in ids]
        sem_list = [desc_map[i].get("sem_desc", "") for i in ids]

        # 结构描述：一般总是存在
        h_struct_text = text_encoder.encode(struct_list)  # [N, d_text]

        # 检查语义描述是否“全空”
        all_sem_empty = all(len(s.strip()) == 0 for s in sem_list)
        if all_sem_empty:
            h_sem_text = None
        else:
            h_sem_text = text_encoder.encode(sem_list)     # [N, d_text]

        return h_struct_text, h_sem_text

    H_text_struct_a, H_text_sem_a = encode_descriptions(desc_a)
    H_text_struct_b, H_text_sem_b = encode_descriptions(desc_b)

    # --------------------------------------------------------
    # (3) GNN 结构编码模型（这里还是占位 Encoder）
    # --------------------------------------------------------
    print("[4/7] Encoding structural features with GNN ...")
    gnn = GNNEncoder(
        in_dim=g1.x.shape[1],
        hidden_dim=128,
        out_dim=128
    ).to(device)

    x1 = torch.tensor(g1.x, dtype=torch.float32, device=device)
    x2 = torch.tensor(g2.x, dtype=torch.float32, device=device)

    # edge_index1/2 视你的封装而定，这里假设是 numpy -> torch
    edge_index1 = torch.tensor(g1.edge_index, dtype=torch.long, device=device)
    edge_index2 = torch.tensor(g2.edge_index, dtype=torch.long, device=device)

    H_struct_a = gnn(x1, edge_index1)  # [N_a, d_struct]
    H_struct_b = gnn(x2, edge_index2)  # [N_b, d_struct]

    # --------------------------------------------------------
    # (4) 多模态融合 F(H_struct, H_text_struct, H_text_sem)
    # --------------------------------------------------------
    print("[5/7] Fusing structural & textual embeddings ...")
    fusion = GatedFusion(
        dim_struct=H_struct_a.shape[1],
        dim_text=H_text_struct_a.shape[1],
        dim_out=128
    ).to(device)

    def fuse_embeddings(
        h_struct: torch.Tensor,
        h_text_struct: torch.Tensor,
        h_text_sem: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        统一的融合入口，自动处理有/无语义文本两种情况。
        逻辑：
        - 只有 struct 文本: 用 struct_desc 文本
        - 只有 sem 文本:    用 sem_desc 文本（理论上很少）
        - 两者都有:         做一个简单融合再喂 Fusion
        - 两者都没有:       退化为只用结构（给个投影层即可）
        """
        h_struct = h_struct.to(device)

        if h_text_struct is None and h_text_sem is None:
            # 完全没有文本，就相当于“只用结构”
            # 你可以直接 return h_struct，也可以加一个 proj 层，这里先直接返回
            return h_struct

        if h_text_struct is not None and h_text_sem is None:
            h_text = h_text_struct.to(device)
        elif h_text_struct is None and h_text_sem is not None:
            h_text = h_text_sem.to(device)
        else:
            # 两种文本都存在，做一个简单平均（你可以改成更复杂的加权）
            h_text = 0.5 * h_text_struct + 0.5 * h_text_sem
            h_text = h_text.to(device)

        return fusion(h_struct, h_text)

    H_a = fuse_three(H_struct_a, H_text_struct_a, H_text_sem_a)  # [N_a, 128]
    H_b = fuse_three(H_struct_b, H_text_struct_b, H_text_sem_b)  # [N_b, 128]

    # --------------------------------------------------------
    # (5) Retrieval (coarse top-k candidates)
    # --------------------------------------------------------
    print("[6/7] Retrieval (coarse top-K candidates) ...")
    retriever = Retriever(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    k = min(topk, num_b)
    topk_indices, _ = retriever.topk_candidates(H_a, H_b, k)  # [N_a, k]
    topk_indices_np = topk_indices.cpu().numpy()

    # --------------------------------------------------------
    # (6) LLM reranking —— 带进度条版本
    # --------------------------------------------------------
    print("[7/7] LLM reranking (pairwise comparison) ...")
    client = OpenAIClient(api_key="YOUR_API_KEY_HERE")
    reranker = LLMReranker(client=client, model=llm_model_name)

    rank_lists_final: List[List[int]] = []

    # tqdm 进度条：num_a 个源节点
    iterator = range(num_a)
    if show_progress:
        iterator = tqdm(iterator, desc="LLM reranking", ncols=100)

    for u in iterator:
        # 拼接源节点描述（结构 + 语义）
        source_desc = (
            desc_a[u]["struct_desc"].strip() + "\n" +
            desc_a[u]["sem_desc"].strip()
        )

        cand_ids = topk_indices_np[u].tolist()
        candidate_descs = [
            (desc_b[v]["struct_desc"].strip() + "\n" + desc_b[v]["sem_desc"].strip())
            for v in cand_ids
        ]

        # LLM pairwise comparison → 新排序
        try:
            reranked = reranker.rerank_one(
                source_desc=source_desc,
                candidate_descs=candidate_descs,
                candidate_ids=cand_ids
            )
        except Exception as e:
            # 避免单个节点错误中断全局：发生问题时 fallback 为原始 topk 顺序
            print(f"[WARN] LLM rerank failed at node {u}, fallback to coarse ranking. Error: {e}")
            reranked = cand_ids

        rank_lists_final.append(reranked)

    rank_lists_final = np.array(rank_lists_final, dtype=np.int64)
    print(">>> Alignment inference finished.")
    print(f"    rank_lists_final shape = {rank_lists_final.shape}")  # [N_a, k]

    return rank_lists_final
