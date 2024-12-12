import os
import argparse
import game_init

def main():
    # Initialize game constants first
    constants = game_init.init()
    
    # Now we can safely import the trainer
    from trainer import LanderTrainer
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Train lunar lander AI')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode without rendering')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to load')
    args = parser.parse_args()

    # Get configuration file path
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-lunar.txt")
    
    # Create trainer instance
    trainer = LanderTrainer(num_landers=20, fast_mode=args.fast)
    
    try:
        # Load checkpoint if specified
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
            
        # Run training for 50 generations
        winner, stats = trainer.run(config_path, n_generations=2000)
        
        if winner:
            print(f"\nBest genome:\n{winner}")
            
            # Save the winner
            with open('best_genome.txt', 'w') as f:
                f.write(str(winner))
                
    except Exception as e:
        print(f"Training error: {e}")
        
    finally:
        if trainer is not None:
            trainer.close()

if __name__ == "__main__":
    main()