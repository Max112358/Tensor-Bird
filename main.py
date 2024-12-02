# main.py
import pygame
import neat
import os
import sys
from constants import *
from bird import Bird
from pipe import Pipe
from background import Background
from game_utils import check_collision, draw_game

pygame.init()

def eval_genomes(genomes, config):
    try:
        birds = []
        nets = []
        ge = []
        best_genome = None
        best_fitness = -float('inf')
        
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(300, 500))
            genome.fitness = 0
            ge.append(genome)
        
        background = Background(SCREEN_WIDTH, SCREEN_HEIGHT)
        pipes = [Pipe(1000 + i * PIPE_SPACING) for i in range(VISIBLE_PIPES)]
        score = 0
        clock = pygame.time.Clock()
        
        run = True
        while run and len(birds) > 0:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    current_best = max(ge, key=lambda x: x.fitness)
                    print('\nBest genome at exit:\n{!s}'.format(current_best))
                    print(f'Final fitness: {current_best.fitness}')
                    pygame.quit()
                    sys.exit()
            
            # Move the parallax background
            background.move()
            
            pipe_ind = 0
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                    pipe_ind = 1
            
            for x, bird in enumerate(birds):
                bird.move()
                ge[x].fitness += 0.1
                
                output = nets[x].activate((
                    bird.y,
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom_y)
                ))
                
                if output[0] > 0.5:
                    bird.jump()
                
                if ge[x].fitness > best_fitness:
                    best_fitness = ge[x].fitness
                    best_genome = ge[x]
            
            for pipe in pipes:
                pipe.move()
                
                for x, bird in enumerate(birds):
                    if check_collision(bird, pipe):
                        ge[x].fitness -= 1
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                    elif not pipe.passed and bird.x > pipe.x + PIPE_WIDTH:
                        pipe.passed = True
                        score += 1
            
            while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))
            
            draw_game(SCREEN, background, pipes, birds, score)
        
        return best_genome
        
    except pygame.error:
        sys.exit()

def run_neat(config_path):
    try:
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        
        winner = pop.run(eval_genomes, 50)
        print('\nBest genome:\n{!s}'.format(winner))
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except SystemExit:
        print("\nTraining terminated")

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)