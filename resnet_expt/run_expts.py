import argparse
import os
import torch
import torch.distributed as dist
from cifar_trainer import NetworkTrainer

def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed CIFAR-10 Gradient Noise Tracking')
    parser.add_argument('--optim', type=str, required=True, choices=['adam', 'sgd'])
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if args.optim == 'adam':
        lr, wd, mom, beta2, epsilon = 0.001, 0.0001, 0.9, 0.999, 1e-08
        mom_args = {'beta1': mom, 'beta2': beta2, 'epsilon': epsilon}
    elif args.optim == 'sgd':
        lr, wd, mom = 0.1, 0.0001, 0.9
        mom_args = {'momentum': mom}
    else: 
        raise NotImplementedError

    if local_rank == 0:
        world_size = dist.get_world_size()
        print(f"\n[DDP] Running on {world_size} GPUs")
        print(f"[Config] Optim: {args.optim} | LR: {lr} | WD: {wd} | Momentum: {mom}\n")

    trainer = NetworkTrainer(
        optim_name=args.optim,
        num_repeats=3,
        lr=lr, wd=wd, 
        device=device, 
        local_rank=local_rank,
        **mom_args
    )
    
    trainer.train_repeatedly()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()