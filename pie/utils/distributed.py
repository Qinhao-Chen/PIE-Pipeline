import os
import torch
import torch.distributed as dist

def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def init_distributed_if_needed():
    """Initializes the distributed process group if variables are present."""
    if dist.is_available() and ("RANK" in os.environ or "LOCAL_RANK" in os.environ):
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        try:
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        except Exception:
            pass

def get_world_size() -> int:
    return dist.get_world_size() if dist_is_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if dist_is_initialized() else 0

def barrier():
    if dist_is_initialized(): dist.barrier()