import pathlib 
import pandas as pd 


class DataLoader:
    """Loads the necessary data for analysis. 

    Example:
        loader = DataLoader()
        exp_df = loader.get_default_exp_df()
    """

    def __init__(self):
        self.mean_exp_sets_df = pd.read_csv("./data/mean_exp_sets.csv")
        self.grouping_tuple = ['machine', 'workload', 'queue_size', 'thread_count', 'iat_scale']

        # default params 
        self.default_queue_size = 128 
        self.default_thread_count = 16 
        self.default_iat_scale = 1 
        self.default_machine = 'c220g1'


    def get_default_exp_df(self):
        """Filters the original df to contain experiment matching the default parameters only. 

        Returns:
            A DataFrame containing outputs from experiments matching the default parameters only. 
        """
        
        return self.mean_exp_sets_df[(self.mean_exp_sets_df['queue_size']==self.default_queue_size) &
                                        (self.mean_exp_sets_df['thread_count']==self.default_thread_count) &
                                        (self.mean_exp_sets_df['iat_scale']==self.default_iat_scale) &
                                        (self.mean_exp_sets_df['machine']==self.default_machine)]