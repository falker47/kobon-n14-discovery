import numpy as np
import matplotlib.pyplot as plt
import os

def generate_fitness_data():
    epochs = np.arange(0, 501)
    
    # Standard GA
    # Rapid climb to 53 by epoch 100, then flatlines
    std_ga = np.zeros_like(epochs, dtype=float)
    std_ga[:101] = 30 + 23 * (1 - np.exp(-0.05 * np.arange(101)))
    std_ga[101:] = std_ga[100]
    
    # Taboo Search GA
    # Matches Standard GA up to epoch 150
    # At epoch 150, fitness drops by -15 (to 38)
    # Then climbs to 54 by epoch 450
    ts_ga = np.copy(std_ga)
    ts_ga[150] = std_ga[149] # Transition
    
    # Penalty drop
    drop_val = std_ga[150] - 15
    ts_ga[150:] = 38 # Start from the drop
    
    # Recovery and climb to global optimum
    climb_epochs = np.arange(150, 501)
    # Sigmoid-like climb from 38 to 54
    # normalized distance from 150 to 500
    progress = (climb_epochs - 150) / (450 - 150)
    climb = 38 + (54 - 38) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
    
    # Fine tune to hit 54 exactly at 450 and stay there
    ts_ga[150:] = np.piecewise(climb_epochs, 
                                [climb_epochs < 450, climb_epochs >= 450],
                                [lambda x: 38 + (54 - 38) * (1 / (1 + np.exp(-0.02 * (x - 300)))), 
                                 54.0])
    
    return epochs, std_ga, ts_ga

def plot_fitness_landscape():
    epochs, std_ga, ts_ga = generate_fitness_data()
    
    # Set academic style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot data
    ax.plot(epochs, std_ga, color='gray', linestyle='--', label='Standard Genetic Algorithm')
    ax.plot(epochs, ts_ga, color='#003366', label='Vectorial Taboo Search (VTS)')
    
    # Horizontal lines for Local and Global optima
    ax.axhline(y=53, color='black', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(505, 52.8, r'$K=53$ (Local Opt.)', verticalalignment='center')
    
    ax.axhline(y=54, color='black', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(505, 54.2, r'$K=54$ (Global Opt.)', verticalalignment='center')
    
    # Annotation for Taboo Penalty
    ax.annotate('Taboo Penalty Applied (-15)', 
                xy=(150, 38), 
                xytext=(180, 42),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=9)
    
    # Styling
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Fitness (K-value)')
    # ax.set_title('Fitness Trajectory: Standard GA vs. Vectorial Taboo Search') # Usually avoided in academic plots if caption is used
    ax.set_ylim(30, 56)
    ax.set_xlim(0, 500)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', frameon=True)
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    # Export
    plt.tight_layout()
    plt.savefig('assets/fitness_landscape.pdf', format='pdf', dpi=300)
    plt.savefig('assets/fitness_landscape.png', format='png', dpi=300)
    print("Files saved: assets/fitness_landscape.pdf, assets/fitness_landscape.png")

if __name__ == "__main__":
    plot_fitness_landscape()
