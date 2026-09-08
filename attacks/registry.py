from core.types import Support

# Adaptations that run. Names carry the `_adapted` suffix so an original paper
# name is never read as a claim that the original method was reproduced.
IMPLEMENTED = {
    "dlg_adapted": "Squared L2 matching + L-BFGS; VLM soft targets and unknown lengths",
    "ig_adapted": "Global cosine matching + TV + signed image gradients with Adam",
    "april_adapted": "Squared L2 + positional-gradient cosine; structural eligibility required",
    "gradvit_adapted": "Layer L2 + BN and patch priors + two-stage scheduler; no registration ensemble",
    "gi_dqa_adapted": "Layer MSE+cosine + document priors with early weighting; no template",
    "random": "No-update random-image/random-token negative control",
    "prior_only": "Image smoothness and optional public text prior; no observed update used",
}

# Accepted by the factory but returning `not_implemented`. Keep in sync with the
# `UnimplementedAttacker.reason` of the corresponding module.
UNIMPLEMENTED = {
    "april": "Closed-form APRIL is not ported; april_adapted is an optimization baseline",
    "dager": "Token span recovery is not ported",
    "embedding_recovery": "Image-token recovery and frozen visual feature inversion are not implemented",
    "idlg": "Classifier label-sign inference is not validated for autoregressive VLM targets",
    "decepticons": "Malicious transformer server requires a separate active-server protocol",
    "imprint": "Model-modifying imprint attack requires a separate active-server protocol",
    "mmgia": "Applicability review only; fused-model transfer not implemented",
}

# Structurally out of scope for this protocol rather than merely unwritten.
NOT_APPLICABLE = {
    "gi_dqa_original": "The no-template VLM protocol cannot reproduce the original template attack",
}

METHODS = {**IMPLEMENTED, **UNIMPLEMENTED, **NOT_APPLICABLE}


def supports(method, adapter, observation):
    if method not in METHODS:
        return Support("not_implemented", f"Unknown method {method}; original names are not adaptation aliases")
    if method in UNIMPLEMENTED:
        return Support("not_implemented", UNIMPLEMENTED[method])
    if method in NOT_APPLICABLE:
        return Support("not_applicable", NOT_APPLICABLE[method])
    if method == "april_adapted":
        names = adapter.position_gradient_names
        if observation.training.observation != "gradient":
            return Support("not_applicable", "APRIL positional derivative premise does not hold for multistep deltas")
        if not names or not all(n in observation.tensors for n in names):
            return Support("not_applicable", "Required visual position embedding gradients are not uploaded")
    return Support("supported")
