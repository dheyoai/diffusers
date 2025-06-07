import torch

def optimized_scale(positive_flat, negative_flat):
    bs = positive_flat.shape[0]
    positive_flat = positive_flat.view(bs, -1)
    negative_flat = negative_flat.view(bs, -1)
    
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
    st_star = dot_product / squared_norm
    return st_star