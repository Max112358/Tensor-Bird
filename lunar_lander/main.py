import os
from trainer import LanderTrainer

def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-lunar.txt")
    
    trainer = LanderTrainer(num_landers=20)
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