import json
import os

class ScoreTracker:
    def __init__(self, filename="scores.json"):
        self.filename = filename
        self.high_score = 0
        self.all_scores = []
        self.load_scores()
    
    def load_scores(self):
        """Load scores from file if it exists"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.high_score = data.get('high_score', 0)
                    self.all_scores = data.get('scores', [])
            except:
                print("Error loading scores file")
    
    def save_scores(self):
        """Save scores to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump({
                    'high_score': self.high_score,
                    'scores': self.all_scores
                }, f)
        except:
            print("Error saving scores file")
    
    def add_score(self, score):
        """Add a new score and update high score if necessary"""
        self.all_scores.append(score)
        if score > self.high_score:
            self.high_score = score
        self.save_scores()
    
    def get_high_score(self):
        return self.high_score
    
    def get_recent_scores(self, n=5):
        """Get the n most recent scores"""
        return self.all_scores[-n:]