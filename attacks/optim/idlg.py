from attacks.base import UnimplementedAttacker


class IDLGAttacker(UnimplementedAttacker):
    reason = "Classifier label-sign inference has not been validated for autoregressive VLM targets."
