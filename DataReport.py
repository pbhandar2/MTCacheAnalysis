import matplotlib.pyplot as plt
import pandas as pd 

from DataLoader import DataLoader

class DataReport:
    """Generates a set of plots that show the distribution of 
    different features in the dataset. 

    Example:
        report = DataReport()
        report_plot_basic()
    """

    def __init__(self):
        self.loader = DataLoader()

    
    def plot_basic(self, output_path="./data/basic_hist.png"):
        """Plots histogram of performance features seen in our experiments. 
        """
        tier2_performance_df = self.loader.get_tier2_performance_df()
        
        plt.figure()
        tier2_performance_df[['t1_hit_rate', 't2_hit_rate', 't1_size', 't2_size', 'percent_diff', 't2_t1_size_ratio']].hist()
        plt.tight_layout()
        plt.savefig(output_path)
        

report = DataReport()
report.plot_basic()