import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def create_and_save_plot(target_genre, bar_labels, bar_heights, colors, normalized_weights, chance, n_coeffs, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(f'Spectral Probing Analysis for Genre: {target_genre}', fontsize=16)

    # Left Panel
    ax1.bar(bar_labels, bar_heights, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=chance, color='r', linestyle='--', label=f'Chance ({chance:.2f})')
    ax1.set_title('Probe Performance')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Right Panel
    x = np.arange(len(normalized_weights))
    ax2.plot(x, normalized_weights, color='black', linewidth=1)

    ax2.set_ylabel('Learned Weight')
    ax2.set_xlabel('Frequency Coefficient')
    ax2.set_xticks([0, n_coeffs//2, n_coeffs-1])
    ax2.set_xticklabels(['L', 'M', 'H'])
    ax2.set_ylim(0, 1)
    ax2.set_title('Spectral Profile')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    clean_genre_name = target_genre.replace(' ', '_').replace('/', '_')
    filename = f'spectral_summary_{clean_genre_name}.png'
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f'Plot for {target_genre} saved to {save_path}')
