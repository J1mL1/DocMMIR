import torch
from torch import Tensor
import math

def calculate_ranking(logits: torch.Tensor) -> tuple[Tensor, Tensor]:
    """
    Calculate the ranking of each correct item within the list of candidates for a retrieval task.

    @args:
        logits (torch.Tensor): Similarity scores (logits) between queries and candidates.
            Shape should be [batch_size, num_candidates], where each row corresponds to a query,
            and each column corresponds to a candidate.

    @returns:
        target_rank (torch.Tensor): Rank position of the correct item for each query.
        ranking (torch.Tensor): Sorted indices of candidates based on similarity to each query.
    """
    # Create labels where each query's correct match is assumed to be at the same index
    labels = torch.arange(logits.size(0)).to(logits.device)

    # Sort the scores in descending order (highest similarity first)
    _, ranking = torch.sort(logits, descending=True)

    # Find the rank of the correct item (position where the label matches the ranking)
    target_rank = (ranking == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1

    return target_rank, ranking


def calculate_ranking_global(full_scores: torch.Tensor, correct_indices: torch.Tensor) -> torch.Tensor:
    """
    For each query, compute the rank (1-indexed) of the correct document.

    @args:
        full_scores: Tensor of shape [B, total_docs] with similarity scores.
        correct_indices: Tensor of shape [B] with the correct global document index for each query.

    @returns:
        Tensor of shape [B] with rank (1-indexed) of the correct document for each query.
    """
    # For each query, sort the scores in descending order and get sorted indices.
    _, sorted_indices = torch.sort(full_scores, descending=True, dim=1)  # [B, total_docs]
    B = full_scores.size(0)
    ranks = torch.empty(B, device=full_scores.device, dtype=torch.float)
    for i in range(B):
        correct_index = correct_indices[i].item()
        # Find where the correct_index appears in sorted_indices[i]
        pos = (sorted_indices[i] == correct_index).nonzero(as_tuple=False)
        if pos.numel() == 0:
            rank = full_scores.size(1)  # worst rank if not found
        else:
            rank = pos.item() + 1  # convert to 1-indexed rank
        ranks[i] = rank
    return ranks


def calculate_metrics(target_rank: torch.Tensor) -> (float, float, float, float):
    """
    Calculate retrieval metrics based on the rank of the correct item for each query.

    @args:
        target_rank (torch.Tensor): Tensor containing the rank position of the correct item for each query.
                                    Shape should be [batch_size], where each element is the rank of the correct item.

    @returns:
        MRR (float): Mean Reciprocal Rank, calculated as the average of the reciprocal ranks of the correct items.
        hit1 (float): Hit@1, the proportion of queries for which the correct item is ranked at position 1.
        hit3 (float): Hit@3, the proportion of queries for which the correct item is ranked within the top 3 positions.
        hit10 (float): Hit@10, the proportion of queries for which the correct item is ranked within the top 10 positions.
    """
    target_rank = target_rank.float()
    mrr = (1.0 / target_rank).mean().item()

    # MRR@10: only count reciprocal rank if target_rank <= 10, else 0
    mrr_at_10 = torch.where(target_rank <= 10, 1.0 / target_rank, torch.zeros_like(target_rank)).mean().item()

    # Hits
    hit1 = (target_rank <= 1).float().mean().item()
    hit3 = (target_rank <= 3).float().mean().item()
    hit10 = (target_rank <= 10).float().mean().item()

    # nDCG@10
    def dcg(rank):
        return 1.0 / math.log2(rank + 1)

    ndcg_vals = torch.where(
        target_rank <= 10,
        torch.tensor([dcg(r.item()) for r in target_rank], device=target_rank.device),
        torch.zeros_like(target_rank)
    )
    idcg = dcg(1)  # ideal rank is always 1
    ndcg_at_10 = (ndcg_vals / idcg).mean().item()

    return {
        "mrr": round(mrr, 4),
        "mrr@10": round(mrr_at_10, 4),
        "hit@1": round(hit1, 4),
        "hit@3": round(hit3, 4),
        "hit@10": round(hit10, 4),
        "ndcg@10": round(ndcg_at_10, 4),
    }
