import csv
import shutil
from pathlib import Path
from contextlib import nullcontext
from typing import Dict # 引入类型注解

import torch
import torch.distributed as dist

from .utils import single_step, get_batch, get_gradient, save_checkpoint, save_worker_state
from .base import eval_and_log

def sample_gradient(layer_types, cfg, device, noise_reader, model, type_ctx, distributed_backend, sample_per_rank=128) -> Dict[str, torch.Tensor]:

    sum_grads: Dict[str, torch.Tensor] = {}
    sum_sq_grads: Dict[str, torch.Tensor] = {}
    
    batch_count = noise_reader.num_batches()
    batch_per_rank = batch_count // dist.get_world_size()
    keep_grad = min(sample_per_rank, batch_per_rank)

    for i in range(keep_grad):
        model.zero_grad(set_to_none=True)

        with model.no_sync():
            data = get_batch(noise_reader, device=device)
            x, y = data

            with type_ctx:
                loss = model(x, targets=y, moe=cfg.moe)["loss"]
            loss.backward()

        with torch.no_grad():
            for lt in layer_types:
                current_grad = get_gradient(lt, model)
                if lt not in sum_grads:
                    sum_grads[lt] = torch.zeros_like(current_grad)
                    sum_sq_grads[lt] = torch.zeros_like(current_grad)

                sum_grads[lt].add_(current_grad)
                sum_sq_grads[lt].add_(current_grad.pow(2))

    total_batches = keep_grad * dist.get_world_size()
    variances: Dict[str, torch.Tensor] = {}

    for lt in layer_types:
        dist.all_reduce(sum_grads[lt], op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_sq_grads[lt], op=dist.ReduceOp.SUM)
        
        sum_grad_f64 = sum_grads[lt].to(torch.float64)
        sum_sq_grad_f64 = sum_sq_grads[lt].to(torch.float64)
        
        mean_grad_f64 = sum_grad_f64 / total_batches
        mean_sq_grad_f64 = sum_sq_grad_f64 / total_batches
        
        variance_f64 = torch.clamp(mean_sq_grad_f64 - mean_grad_f64.pow(2), min=0.0)
        variances[lt] = variance_f64.to(torch.float32).cpu()
    
    model.zero_grad(set_to_none=True)
    return variances

def compute_noise_stats(variance_tensor):
    M = variance_tensor.numel()
    sigma = torch.sqrt(variance_tensor)
    l1 = sigma.sum().item()
    l2_sq = variance_tensor.sum().item() 
    
    sparsity = (l1 ** 2) / (M * l2_sq) if l2_sq > 0 else 0.0
    return sparsity

def eval_noise(model, opt, datareaders, scheduler, cfg, exp_dir, distributed_backend, noise_reader, save_cnt, sample_itr, sample_per_rank=32):
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    
    layer_types = ["all"]

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16)
    else:
        type_ctx = nullcontext()

    log_dir = Path(exp_dir / "noise")
    if log_dir.exists() and distributed_backend.is_master_process():
        shutil.rmtree(log_dir)
    dist.barrier()
    log_dir.mkdir(parents=True, exist_ok=True)

    is_master = distributed_backend.is_master_process()
    total_steps = cfg.iterations

    log_files = {
        "start": log_dir / "density_start.csv",
        "mid": log_dir / "density_mid.csv",
        "end": log_dir / "density_end.csv"
    }

    if is_master:
        for phase, filepath in log_files.items():
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration'] + layer_types)
        print(f"> Training started. Tracking layers: {layer_types}")

    def get_phase(step):
        if step < sample_itr: return "start"
        elif (total_steps // 2) <= step < (total_steps // 2) + sample_itr: return "mid"
        elif total_steps - sample_itr <= step < total_steps: return "end"
        return None

    device = cfg.device
    saved_phases = set()
    for itr_step in range(total_steps):
        phase = get_phase(itr_step)
        
        if phase is not None:
            if phase not in saved_phases:
                if is_master:
                    ckpt_path = log_dir / f"model_weights_{phase}.pt"
                    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(state_dict, ckpt_path)
                    print(f"[{phase.upper()}] Weights saved to {ckpt_path}")
                saved_phases.add(phase)

            model.eval()
            variances_dict = sample_gradient(
                layer_types=layer_types, cfg=cfg, device=device, noise_reader=noise_reader,
                model=model, distributed_backend=distributed_backend,
                type_ctx=type_ctx, sample_per_rank=sample_per_rank
            )

            row_data = [itr_step]
            print_strs = []
            
            for lt in layer_types:
                sp = compute_noise_stats(variances_dict[lt])
                row_data.append(sp)
                print_strs.append(f"{lt}: {sp:.4f}")

            if is_master:
                with open(log_files[phase], 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                
                if itr_step % 10 == 0:
                    print(f"[{phase.upper()}] Step {itr_step} | " + " | ".join(print_strs), flush=True)

            del variances_dict

        # 全速前进
        loss = single_step(model, opt, scheduler, train_reader, type_ctx, distributed_backend, cfg)

        if itr_step % 100 == 0:
            if is_master:
                print(f"> Training Loss at {itr_step} : {loss:.4f}")

    if is_master:
        print("> Training and noise tracking completed successfully.")
        
    return