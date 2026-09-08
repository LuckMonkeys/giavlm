"""Differentiable named-update matching objectives shared by VLM adaptations."""
def matching_loss(predicted, observed, kind):
    if kind not in {"l2", "cosine", "tag", "layer_l2", "combined"}:
        raise ValueError(f"Unknown matching objective: {kind}")
    if not observed:
        raise ValueError("Named updates cannot be empty")
    if set(predicted) != set(observed):
        raise ValueError("Candidate and observed parameter sets differ")
    pairs = [(predicted[k].float(), observed[k].float()) for k in sorted(observed)]
    if any(x.shape != y.shape for x, y in pairs):
        raise ValueError("Candidate and observed update shapes differ")
    device = pairs[0][0].device
    pairs = [(x, y.to(x.device)) for x, y in pairs]
    def reduce(values):
        return sum(value.to(device) for value in values)
    if kind == "cosine":
        dot = reduce((x * y).sum() for x, y in pairs)
        nx = reduce(x.square().sum() for x, _ in pairs)
        ny = reduce(y.square().sum() for _, y in pairs)
        if ny.detach().item() == 0:
            return nx
        return 1 - dot / (nx.sqrt() * ny.sqrt()).clamp_min(1e-20)
    if kind == "tag":
        return reduce((x - y).square().sum() + 0.01 * (x - y).abs().sum() for x, y in pairs)
    if kind == "layer_l2":
        return reduce((x - y).norm() for x, y in pairs)
    if kind == "combined":
        terms = []
        for x, y in pairs:
            mse = (x - y).square().mean()
            if y.detach().square().sum().item() > 0:
                mse = mse + 1 - (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-20)
            terms.append(mse)
        return reduce(terms)
    return reduce((x - y).square().sum() for x, y in pairs)
