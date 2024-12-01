import argparse
from fast_learning import FastLearningFlappyBird

def main():
    parser = argparse.ArgumentParser(description='Run Flappy Bird AI training')
    parser.add_argument('--mode', choices=['fast', 'visual'], default='fast',
                      help='Training mode: fast (no graphics) or visual (with display)')
    parser.add_argument('--games', type=int, default=10000,
                      help='Number of games to train (fast mode only)')
    parser.add_argument('--training-speed', type=int, default=20,
                      help='Number of games between training updates (fast mode only)')
    parser.add_argument('--display-interval', type=int, default=100,
                      help='Number of games between progress updates (fast mode only)')
    parser.add_argument('--model-path', default='saved_model',
                      help='Path to save/load the model')
    
    args = parser.parse_args()
    
    try:
        print(f"Starting training for {args.games} games")
        print("Press Ctrl+C to stop training")
        
        ai_game = FastLearningFlappyBird(
            model_path=args.model_path,
            training_speed=args.training_speed
        )
        ai_game.run_fast_training(
            num_games=args.games,
            display_interval=args.display_interval
        )
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        print("Training completed")

if __name__ == "__main__":
    main()
    
    #python run_flappy.py --games 1000 --training-speed 50 --display-interval 10