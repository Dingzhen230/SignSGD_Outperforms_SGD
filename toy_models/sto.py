import torch
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['axes.unicode_minus'] = False 

def run_experiment(d, lr, optim_type, steps=3000, seeds=50):
    losses = np.zeros((seeds, steps))
    
    torch.manual_seed(42)  
    x = torch.randn(seeds, d)
    
    for t in range(steps):
        loss = 0.5 * torch.sum(x**2, dim=1)
        losses[:, t] = loss.numpy()
        
        g = x.clone()

        noise = torch.randn(seeds) * 100.0
        g[:, 0] += noise
        
        if optim_type == 'sgd':
            x -= lr * g
        elif optim_type == 'signsgd':
            x -= lr * torch.sign(g)
            
    return losses

def main():
    steps = 3000
    seeds = 42
    
    lr_sgd = 0.001
    lr_signsgd = 0.01
    
    
    for d in [100]:
        plt.figure(figsize=(9, 6))
        
        for optim_name, color, label, lr in zip(
            ['sgd', 'signsgd'], 
            ['blue', 'green'], 
            ['SGD', 'signSGD'],
            [lr_sgd, lr_signsgd]
        ):
            losses = run_experiment(d, lr, optim_name, steps=steps, seeds=seeds)
            
            mean_loss = np.mean(losses, axis=0)
            std_loss = np.std(losses, axis=0)
            steps_arr = np.arange(steps)
            
            plt.plot(steps_arr, mean_loss, label=f'{label}', color=color)
            lower_bound = np.percentile(losses, 10, axis=0)
            upper_bound = np.percentile(losses, 90, axis=0)

            plt.fill_between(steps_arr, lower_bound, upper_bound, color=color, alpha=0.3)
        
        if d == 100:
            plt.ylim(-2, 60)
        else:
            plt.ylim(-2, d * 0.6)
            
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filename = f'toy_sto_d={d}.pdf'
        plt.savefig(filename, dpi=512, bbox_inches='tight')
        plt.close()
        
        print(f"saved to: {filename}")

if __name__ == '__main__':
    main()