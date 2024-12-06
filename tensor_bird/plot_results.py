import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results():
    # Load the results
    with open('training_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract generations data
    generations = [r['generations'] for r in results]
    
    # Calculate statistics
    avg_generations = np.mean(generations)
    median_generations = np.median(generations)
    std_generations = np.std(generations)
    
    # Create the histogram
    plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(generations, bins=10, edgecolor='black')
    
    # Add a line for the mean
    plt.axvline(avg_generations, color='r', linestyle='--', label=f'Mean: {avg_generations:.1f}')
    plt.axvline(median_generations, color='g', linestyle='--', label=f'Median: {median_generations:.1f}')
    
    # Customize the plot
    plt.title('Distribution of Generations Required to Reach Target Fitness\n' + 
             f'Mean: {avg_generations:.1f} | Median: {median_generations:.1f} | Std Dev: {std_generations:.1f}')
    plt.xlabel('Number of Generations')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with detailed statistics
    stats_text = (f'Number of Sessions: {len(generations)}\n'
                 f'Mean: {avg_generations:.1f}\n'
                 f'Median: {median_generations:.1f}\n'
                 f'Std Dev: {std_generations:.1f}\n'
                 f'Min: {min(generations)}\n'
                 f'Max: {max(generations)}')
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('training_results_histogram.png')
    plt.close()

if __name__ == "__main__":
    plot_training_results()
    print("Histogram has been saved as 'training_results_histogram.png'")