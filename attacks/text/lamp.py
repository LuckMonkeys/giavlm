"""LAMP-style discrete text proposals, scored only with public update/prior data."""
import torch


def permutation_step(engine, candidate):
    from attacks.optim.engine import BudgetExhausted
    """LAMP-style swap/move proposals scored with update residual and public LM."""
    score = engine.discrete_score(candidate)
    for name, private in [("questions", candidate.private_q), ("targets", candidate.private_y)]:
        if not private:
            continue
        values = getattr(candidate, name)
        original = values.detach().clone()
        if values.shape[1] < 3:
            continue
        # The final public EOS is fixed. All other positions, including candidate EOS, can move.
        for proposal in range(2):
            with torch.no_grad():
                perm = torch.arange(values.shape[1], device=values.device)
                indices = torch.randperm(values.shape[1] - 1, device=values.device)[:2]
                i, j = sorted(indices.tolist())
                if proposal == 0:
                    perm[i], perm[j] = perm[j].clone(), perm[i].clone()
                else:
                    perm[i:j + 1] = perm[i:j + 1].roll(1)
                values.copy_(original[:, perm])
            try:
                new_score = engine.discrete_score(candidate)
            except BudgetExhausted:
                with torch.no_grad():
                    values.copy_(original)
                raise
            if new_score < score:
                score, original = new_score, values.detach().clone()
            with torch.no_grad():
                values.copy_(original)
    engine.remember(candidate, score)
