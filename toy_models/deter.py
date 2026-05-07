import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False 

def run_trajectory(d, lr, optim_type, x_init, L, steps=3000):
    x = x_init.clone()
    losses = np.zeros(steps)
    
    for t in range(steps):
        loss = 0.5 * torch.sum(L * (x**2))
        losses[t] = loss.item()
        
        g = L * x
        
        if optim_type == 'sgd':
            x -= lr * g
        elif optim_type == 'signsgd':
            x -= lr * torch.sign(g)
            
    return losses

def main():
    steps = 3000
    d = 5000
    
    candidate_lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    
    x_init = torch.randn(d)
    
    L = torch.ones(d)
    L[0] = 1000
    x_init[0] /= L[0]
    
    best_lrs = {}
    best_trajectories = {}
    
    for optim_name in ['sgd', 'signsgd']:
        best_lr = None
        best_loss = float('inf')
        best_traj = None
        
        for lr in candidate_lrs:
            traj = run_trajectory(d, lr, optim_name, x_init, L, steps=steps)
            final_loss = traj[-1]
            
            if not np.isnan(final_loss) and final_loss < best_loss and final_loss <= traj[0]:
                best_loss = final_loss
                best_lr = lr
                best_traj = traj
                
        best_lrs[optim_name] = best_lr
        best_trajectories[optim_name] = best_traj
        print(f"[{optim_name.upper()}] optimal lr: {best_lr}, final Loss: {best_loss:.6f}")

    plt.figure(figsize=(9, 6))
    
    plt.plot(np.arange(steps), best_trajectories['sgd'], label=f"SGD", color='blue', linewidth=2)
    plt.plot(np.arange(steps), best_trajectories['signsgd'], label=f"signSGD", color='green', linewidth=2)
    
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filename = f'toy_deter_d={d}.pdf'
    plt.savefig(filename, dpi=512, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to: {filename}")

if __name__ == '__main__':
    main()