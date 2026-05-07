import torch
import torch.distributed as dist
import random

def sample_cifar_gradient(dataloader, model, criterion, device):
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    track_idx = random.randint(0, total_params - 1)
    grad_samples = []

    sum_grad = None
    sum_sq_grad = None
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        model.zero_grad(set_to_none=True)
        
        with model.no_sync():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        with torch.no_grad():
            grads = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.requires_grad])
            
            grad_samples.append(grads[track_idx].item())

            if sum_grad is None:
                sum_grad = torch.zeros_like(grads)
                sum_sq_grad = torch.zeros_like(grads)

            sum_grad.add_(grads)
            sum_sq_grad.add_(grads.pow(2))
            total_samples += 1

    dist.all_reduce(sum_grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_sq_grad, op=dist.ReduceOp.SUM)
    
    total_samples_tensor = torch.tensor([total_samples], device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    global_total_samples = total_samples_tensor.item()

    sum_grad_f64 = sum_grad.to(torch.float64)
    sum_sq_grad_f64 = sum_sq_grad.to(torch.float64)
    
    mean_f64 = sum_grad_f64 / global_total_samples
    mean_sq_f64 = sum_sq_grad_f64 / global_total_samples
    
    variance_f64 = torch.clamp(mean_sq_f64 - mean_f64.pow(2), min=0.0)
    
    model.zero_grad(set_to_none=True)
    
    return mean_f64.to(torch.float32), variance_f64.to(torch.float32), grad_samples