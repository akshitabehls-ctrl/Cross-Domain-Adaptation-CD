import torch
import torch.nn.functional as F

@torch.no_grad()
def sinkhorn_knopp(out, epsilon=0.05, iterations=3, prior=None):
    """
    Applies the Sinkhorn-Knopp algorithm to softly assign probabilities
    with an arbitrary class prior over the batch/spatial dimensions.
    
    Args:
        out: logits of shape [N, K] (e.g., N=Batch*H*W, K=classes)
        epsilon: temperature for entropic regularization
        iterations: number of sinkhorn iterations
        prior: list of prior probabilities for each class, summing to 1 (e.g. [0.95, 0.05])
        
    Returns:
        Q: soft assignments of shape [N, K]
    """
    Q = torch.exp(out / epsilon).clone()
    
    B = Q.shape[0]
    K = Q.shape[1]

    if prior is None:
        prior = torch.ones(1, K, device=out.device) / K
    else:
        prior = torch.tensor(prior, device=out.device).view(1, K)

    # Make the matrix sum to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(iterations):
        # normalize each row: total weight per row must be 1/B
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= B

        # normalize each column: match the desired prior for classes
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q *= prior

    Q *= B
    return Q
