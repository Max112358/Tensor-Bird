import os
import argparse
import game_init
import neat
import pickle

def create_visualize_module():
    """Create the visualize.py module if it doesn't exist"""
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

def create_dirs():
    """Create necessary directories for outputs"""
    directories = ['checkpoints', 'outputs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_neat_config_value(config_path: str, param_name: str, section: str = 'NEAT') -> int:
    """
    Read a specific parameter from the NEAT config file
    Returns the parameter value or a default if not found
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)
    try:
        return int(config[section][param_name])
    except (KeyError, ValueError):
        print(f"Warning: Could not read {param_name} from config file")
        return 20  # Default fallback value

def main():
    # Create visualize module if it doesn't exist
    if not os.path.exists('visualize.py'):
        create_visualize_module()
    
    # Get configuration file path
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-lunar.txt")
    
    # Read population size from NEAT config
    pop_size = get_neat_config_value(config_path, 'pop_size')
    
    # Now we can safely import visualize
    import visualize
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train lunar lander AI')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode without rendering')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to restore from')
    parser.add_argument('--generations', type=int, default=200000, help='Number of generations to train')
    parser.add_argument('--num-landers', type=int, default=pop_size, 
                       help=f'Number of landers to train simultaneously (default: {pop_size} from NEAT config)')
    parser.add_argument('--checkpoint-interval', type=int, default=250, 
                       help='How often to save checkpoints (in generations)')
    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    # Verify checkpoint file if specified
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")

    try:
        # Initialize game constants first
        constants = game_init.init()
        
        # Import trainer here to ensure game_init has been called
        from trainer import LanderTrainer
        
        # Create trainer instance
        trainer = LanderTrainer(
            num_landers=args.num_landers,
            checkpoint_interval=args.checkpoint_interval,
            fast_mode=args.fast
        )
        
        try:
            # Run training
            winner, stats = trainer.run(
                config_path=config_path,
                n_generations=args.generations,
                checkpoint_file=args.checkpoint
            )
            
            if winner:
                print(f"\nBest genome:\n{winner}")
                
                # Save the winner genome
                winner_path = os.path.join('outputs', 'best_genome.pkl')
                print(f"Saving best genome to {winner_path}")
                with open(winner_path, 'wb') as f:
                    pickle.dump(winner, f)
                
                # Visualize the winner network
                try:
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
                    visualize.draw_net(config, winner, True, node_names=node_names,
                                     filename=os.path.join('outputs', 'winner_network'))
                    visualize.plot_stats(stats, ylog=False, view=True,
                                       filename=os.path.join('outputs', 'fitness_history.svg'))
                    visualize.plot_species(stats, view=True,
                                         filename=os.path.join('outputs', 'speciation.svg'))
                except Exception as e:
                    print(f"Error during visualization: {e}")
            else:
                print("\nNo winner found")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {e}")
            raise
            
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
        
    finally:
        if 'trainer' in locals():
            trainer.close()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)