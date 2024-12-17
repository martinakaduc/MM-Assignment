import numpy as np
import matplotlib.pyplot as plt

class EpisodeEvaluator:
    def __init__(self):
        self.metrics = {
            'episode_number': 0,
            'filled_ratio': 0.0,
            'trim_loss': 0.0,
        }
        #Add lists to store history of metrics
        self.history = {
            'episode_numbers': [],  
            'filled_ratios': [],
            'trim_losses': [],
        }
    
    def calculate_metrics(self, observation):
        """Calculate trimloss and fill ratio from observation"""
        total_stocks = len(observation['stocks'])
        used_stocks = 0
        total_used_area = 0
        total_stock_area = 0
        
        for stock in observation['stocks']:
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))
            stock_area = stock_w * stock_h
            used_area = np.sum(stock >= 0)  # Count non-empty cells
            

            # Check if stock is used
            if used_area > 0:
                used_stocks += 1
                total_used_area += used_area
                total_stock_area += stock_area
        
        # Calculate trimloss (ratio of unused area in used stocks)
        trimloss = (total_stock_area - total_used_area) / total_stock_area if total_stock_area > 0 else 0.0
        
        # Calculate fill ratio (ratio of used stocks to total stocks)
        fill_ratio = used_stocks / total_stocks if total_stocks > 0 else 0.0
        
        return trimloss, fill_ratio, used_stocks

    def evaluate_episode(self, observation, episode_data):
        """Calculate comprehensive episode quality score with correct filled ratio"""
        trimloss, fill_ratio, used_stock = self.calculate_metrics(observation)
        
        self.metrics.update({
            'episode_number': episode_data['episode_number'],
            'filled_ratio': fill_ratio,
            'trimloss': trimloss,
        })
        
        # Store in history
        self.history['episode_numbers'].append(episode_data['episode_number'])
        self.history['filled_ratios'].append(fill_ratio)
        self.history['trim_losses'].append(trimloss)
        
        return self.get_summary()
    
    def get_summary(self):
        """Return formatted summary of episode performance"""
        summary = f"\n{'='*20} Episode {self.metrics['episode_number']} Quality Report {'='*20}\n"
        summary += f"Filled Ratio: {self.metrics['filled_ratio']:.3f}\n"
        summary += f"Trimloss: {self.metrics['trimloss']:.3f}\n"
        summary += "="*70
        return summary
    
    def plot_metrics(self):
        """Plot the metrics history"""
        plt.figure(figsize=(12, 5))
        
        # Plot filled ratios
        plt.subplot(1, 2, 1)
        plt.plot(self.history['episode_numbers'], self.history['filled_ratios'], 'b-', label='Filled Ratio')
        plt.title('Filled Ratio vs Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Filled Ratio')
        plt.grid(True)
        plt.legend()
        
        # Plot trim losses
        plt.subplot(1, 2, 2)
        plt.plot(self.history['episode_numbers'], self.history['trim_losses'], 'r-', label='Trim Loss')
        plt.title('Trim Loss vs Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Trim Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cutting_stock_metrics.png')
        plt.close()