import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def coma_select_samples(model_a, model_b, dataset, k=100, batch_size=64, device="cpu"):
    """
    coMA: Co-Training + Model Agreement Disagreement Sampling
    Returns indices of the top-k samples where the two models disagree most.
    """

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    disagreement_scores = []
    all_indices = []

    idx_ptr = 0
    model_a.eval()
    model_b.eval()

    for images, _ in loader:
        images = images.to(device)

        logits_a = model_a(images)
        logits_b = model_b(images)

        probs_a = torch.softmax(logits_a, dim=1)
        probs_b = torch.softmax(logits_b, dim=1)

        pred_a = probs_a.argmax(dim=1)
        pred_b = probs_b.argmax(dim=1)

        # Disagreement mask
        disagree = pred_a != pred_b

        # Confidence difference = uncertainty measure
        conf_gap = (probs_a.max(dim=1).values - probs_b.max(dim=1).values).abs()

        for i in range(len(images)):
            if disagree[i]:
                disagreement_scores.append(float(conf_gap[i].cpu()))
                all_indices.append(idx_ptr + i)

        idx_ptr += len(images)

    # Sort samples by disagreement score (descending)
    if len(all_indices) == 0:
        return []

    sorted_idx = sorted(range(len(all_indices)), key=lambda x: disagreement_scores[x], reverse=True)

    # return top-k original dataset indices
    chosen = [all_indices[i] for i in sorted_idx[:k]]

    return chosen
