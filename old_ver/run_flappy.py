import argparse
from enhanced_game_loop import EnhancedFlappyTrainer

def main():
    parser = argparse.ArgumentParser(description='Run Flappy Bird AI training with enhanced genetics')
    parser.add_argument('--games', type=int, default=10000,
                      help='Number of games to train (fast mode only)')
    parser.add_argument('--display-interval', type=int, default=100,
                      help='Number of games between progress updates')
    parser.add_argument('--model-path', default='saved_model',
                      help='Path to save/load the model')
    
    args = parser.parse_args()
    
    try:
        print(f"Starting enhanced genetic training for {args.games} games")
        print("Press Ctrl+C to stop training")
        
        trainer = EnhancedFlappyTrainer('config-feedforward.txt')
        winner, stats = trainer.run(num_generations=args.games)
        
        # Save the winner
        trainer.save_genome(winner, args.model_path)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        print("Training completed")

if __name__ == "__main__":
    main()