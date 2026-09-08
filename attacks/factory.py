"""Explicit registration. Original paper names never alias VLM adaptations."""
from attacks.base import OptimizationAttacker, UnimplementedAttacker


def create_attacker(adapter, spec):
    name = spec.method
    if name == "dlg_adapted":
        from attacks.optim.dlg import DLGAttacker
        return DLGAttacker(adapter, spec)
    elif name == "ig_adapted":
        from attacks.optim.inverting_gradients import IGAttacker
        return IGAttacker(adapter, spec)
    elif name == "gradvit_adapted":
        from attacks.optim.gradvit import GradViTAttacker
        return GradViTAttacker(adapter, spec)
    elif name == "gi_dqa_adapted":
        from attacks.optim.gidqa import GIDQAAttacker
        return GIDQAAttacker(adapter, spec)
    elif name == "prior_only":
        from attacks.prior_only import PriorOnlyAttacker
        return PriorOnlyAttacker(adapter, spec)
    elif name in {"april_adapted", "random", "gi_dqa_original"}:
        return OptimizationAttacker(adapter, spec)
    elif name == "april":
        from attacks.analytic.april import APRILAttacker
        return APRILAttacker(adapter, spec)
    elif name == "dager":
        from attacks.analytic.dager import DAGERAttacker
        return DAGERAttacker(adapter, spec)
    elif name == "embedding_recovery":
        from attacks.analytic.embedding_recovery import EmbeddingRecoveryAttacker
        return EmbeddingRecoveryAttacker(adapter, spec)
    elif name == "idlg":
        from attacks.optim.idlg import IDLGAttacker
        return IDLGAttacker(adapter, spec)
    elif name == "decepticons":
        from attacks.malicious.decepticons import DecepticonsAttacker
        return DecepticonsAttacker(adapter, spec)
    elif name == "imprint":
        from attacks.malicious.imprint import ImprintAttacker
        return ImprintAttacker(adapter, spec)
    elif name == "mmgia":
        return UnimplementedAttacker(adapter, spec)
    raise ValueError(f"Unknown attack: {name}")


class AttackFactory:
    create_attacker = staticmethod(create_attacker)
