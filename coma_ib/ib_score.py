import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_ib_scores(model, dataset, k=100, batch_size=64, device="cpu"):
    """
    Information Bottleneck uncertainty sampling.
    Select samples with highest entropy (information content).
    """

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    indices = []

    idx_ptr = 0

    for images, _ in loader:
        images = images.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        entropy = -(probs * probs.log()).sum(dim=1)

        for i in range(len(entropy)):
            scores.append(float(entropy[i].cpu()))
            indices.append(idx_ptr + i)

        idx_ptr += len(images)

    # sort entropy descending: top information samples
    sorted_idx = sorted(range(len(indices)), key=lambda x: scores[x], reverse=True)

    chosen = [indices[i] for i in sorted_idx[:k]]
    return chosen
