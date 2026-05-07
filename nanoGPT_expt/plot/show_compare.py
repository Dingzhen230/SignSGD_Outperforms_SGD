import os
import pandas as pd
import matplotlib.pyplot as plt

experiments = {
    "SGD": "Your SGD result folder",
    "SignSGD": "Your SignSGD result folder",
}

metrics = {
    "train_loss": "Train Loss",
    "val_loss": "Validation Loss",  
    "val_acc": "Validation Accuracy"
}

plt.style.use('seaborn-v0_8-whitegrid')

for metric_key, metric_title in metrics.items():
    plt.figure(figsize=(9, 6))
    
    for opt_name, base_path in experiments.items():
        csv_path = os.path.join(base_path, "logs", f"{metric_key}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            if metric_key in ["val_loss", "val_acc"] and len(df) > 0:
                df = df.iloc[:-1]
            
            if 'iteration' in df.columns:
                y_col = df.columns[1] 
                plt.plot(df['iteration'], df[y_col], label=opt_name, linewidth=2)
            else:
                raise ValueError(f"Could not find 'iteration' col in {csv_path}")
        else:
            raise FileNotFoundError(f"could not find file {csv_path}")
            
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel(metric_title, fontsize=18)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_filename = f"{metric_key}_comparison.pdf"
    
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"saved_to: {output_filename}")
    
    plt.close()

print("all pdf file generated!")