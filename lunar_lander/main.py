import os
import argparse
from trainer import LanderTrainer

def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Train lunar lander AI')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode without rendering')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-lunar.txt")
    
    trainer = LanderTrainer(num_landers=20, fast_mode=args.fast)
    try:
        winner, stats = trainer.run(config_path)
        print(f"\nBest genome:\n{winner}")
        
        # Save the winner
        with open('best_genome.txt', 'w') as f:
            f.write(str(winner))
            
    finally:
        trainer.close()

if __name__ == "__main__":
    main()