import neat
import visualize
from typing import Dict, Optional

def visualize_network(config_path: str, genome_path: str):
    """
    Visualize a NEAT neural network's structure
    
    Args:
        config_path: Path to NEAT config file
        genome_path: Path to saved genome file
    """
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Load genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # Define node names for visualization
    node_names = {
        -1: 'velocity_x',
        -2: 'velocity_y', 
        -3: 'angle',
        -4: 'angular_velocity',
        -5: 'dist_to_pad_x',
        -6: 'dist_to_pad_y',
        0: 'left_thruster',
        1: 'main_thruster', 
        2: 'right_thruster'
    }
    
    # Create visualization
    visualize.draw_net(config, genome, True, node_names=node_names)
    
    # If you have statistics saved, you can also visualize:
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

def create_visualize_module():
    """
    Create a visualize.py module with required visualization functions
    """
    visualize_code = '''
import graphviz
import matplotlib.pyplot as plt
import numpy as np

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, best_fitness, 'r-', label="best")

    if ylog:
        plt.gca().set_yscale('symlog')

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if view:
        plt.show()
    plt.savefig(filename)
    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    if view:
        plt.show()
    plt.savefig(filename)
    plt.close()

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for n in genome.nodes.keys():
        if n in inputs or n in outputs:
            continue
        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    return dot
'''
    with open('visualize.py', 'w') as f:
        f.write(visualize_code)

if __name__ == "__main__":
    # First create the visualize module
    create_visualize_module()
    
    # Then use it to visualize a network
    config_path = "config-lunar.txt"  # Path to your NEAT config file
    genome_path = "best_genome.pkl"   # Path to your saved genome
    visualize_network(config_path, genome_path)