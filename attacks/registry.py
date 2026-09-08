from core.types import Support

METHODS = {
    "dlg_adapted": "Squared L2 matching + L-BFGS; VLM soft targets and unknown lengths",
    "ig_adapted": "Global cosine matching + TV + signed image gradients with Adam",
    "april_adapted": "Squared L2 + positional-gradient cosine; structural eligibility required",
    "gradvit_adapted": "Layer L2 + BN and patch priors + two-stage scheduler; no registration ensemble",
    "gi_dqa_adapted": "Layer MSE+cosine + document priors with early weighting; no template",
    "random": "No-update random-image/random-token negative control",
    "prior_only": "Image smoothness and optional public text prior; no observed update used",
    "dager": "Applicability review only; exact text inversion not ported",
    "mmgia": "Applicability review only; fused-model transfer not implemented",
    "gi_dqa_original": "Use the separate legacy document-template protocol",
}


def supports(method, adapter, observation):
    if method not in METHODS:
        return Support("not_implemented", f"Unknown method {method}; original names are not adaptation aliases")
    if method in {"dager", "mmgia"}:
        return Support("not_implemented", METHODS[method])
    if method == "gi_dqa_original":
        return Support("not_applicable", "The no-template VLM protocol cannot reproduce the original template attack")
    if method == "april_adapted":
        names = adapter.position_gradient_names
        if observation.training.observation != "gradient":
            return Support("not_applicable", "APRIL positional derivative premise does not hold for multistep deltas")
        if not names or not all(n in observation.tensors for n in names):
            return Support("not_applicable", "Required visual position embedding gradients are not uploaded")
    return Support("supported")
