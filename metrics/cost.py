"""Resource counters; wall time is not mislabeled as measured GPU utilization."""
import torch


def resource_costs(adapter, seconds, update_evaluations, local_backward_evaluations,
                   prior_evaluations):
    devices = {p.device for p in adapter.parameters() if p.device.type == "cuda"}
    peaks = {str(d): torch.cuda.max_memory_allocated(d) for d in devices}
    return {"seconds": seconds, "wall_hours": seconds / 3600,
            "gpu_count": len(devices), "update_evaluations": update_evaluations,
            "local_backward_evaluations": local_backward_evaluations,
            "prior_evaluations": prior_evaluations,
            "peak_cuda_bytes": peaks.get(str(adapter.device), 0),
            "peak_cuda_bytes_by_device": peaks}
