import torch
import torch.nn.functional as F

@torch.no_grad()
def sinkhorn_knopp(out, epsilon=0.05, iterations=3):
    """
    Applies the Sinkhorn-Knopp algorithm to softly assign probabilities
    ensuring equipartition over the batch/spatial dimensions.
    
    Args:
        out: logits of shape [N, K] (e.g., N=Batch*H*W, K=classes)
        epsilon: temperature for entropic regularization
        iterations: number of sinkhorn iterations
        
    Returns:
        Q: soft assignments of shape [N, K]
    """
    Q = torch.exp(out / epsilon).clone()
    
    B = Q.shape[0]
    K = Q.shape[1]

    # Make the matrix sum to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(iterations):
        # normalize each row: total weight per row must be 1/B
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= B

        # normalize each column: total weight per column must be 1/K
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q /= K

    Q *= B
    return Q
