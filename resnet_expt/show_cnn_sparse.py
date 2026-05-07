import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cnn_sparsity(csv_filepath, output_filename='resnet_sparsity_evolution.pdf'):
    if csv_filepath == "your csv file path":
        raise ValueError("Please change CSV_FILE to your csv file path")
    filepath = Path(csv_filepath)
    
    if not filepath.exists():
        raise ValueError(f"❌ could not find {filepath}")
        
    df = pd.read_csv(filepath)
    
    if 'sparsity_phi' not in df.columns or 'epoch' not in df.columns:
        raise ValueError(f"❌ Error: lack of column namse, exists: {list(df.columns)}")

    epochs = df['epoch']
    raw_sparsity = df['sparsity_phi']

    smoothing_span = 5  
    smoothed_sparsity = raw_sparsity.ewm(span=smoothing_span, adjust=False).mean()
    
    tail_len = max(int(len(raw_sparsity) * 0.1), 5)
    converge_val = raw_sparsity.tail(tail_len).mean()
    
    gaussian_val = 0.6366
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs, raw_sparsity, 
        color="#1f77b4", alpha=0.15, linewidth=1, label="_nolegend_"
    )
    
    plt.plot(
        epochs, smoothed_sparsity, 
        label="ResNet", 
        color="#1f77b4", linewidth=2.5
    )

    plt.axhline(y=gaussian_val, color='gray', linestyle='--', alpha=0.7, label='Standard Gaussian')
    
    plt.axhline(y=converge_val, color='#d62728', linestyle=':', linewidth=2, alpha=0.8, label='Convergence Limit')

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Variance Density $\\phi(\\sigma)$", fontsize=18)
    
    plt.ylim(0, 1.05)
    base_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
    all_ticks = base_ticks + [gaussian_val]
    if abs(converge_val - gaussian_val) > 0.035:
        all_ticks.append(converge_val)
    
    all_ticks.sort()
    plt.yticks(all_ticks)

    ax = plt.gca()
    for label in ax.get_yticklabels():
        tick_val = label.get_position()[1]
        
        if abs(tick_val - converge_val) < 1e-5:
            label.set_color('#d62728')      
            label.set_fontweight('bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=18, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    output_path = "./" + output_filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"✅ file saved to: {output_path}")

if __name__ == "__main__":
    CSV_FILE = "Your csv file path"  
    OUTPUT_PDF = "noise_sparstiy_resnet.pdf"
    
    plot_cnn_sparsity(CSV_FILE, OUTPUT_PDF)