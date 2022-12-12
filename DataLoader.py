import pandas as pd 


class DataLoader:
    """Loads the necessary data for analysis. Each row of the data is an aggregated output
    from 3 iterations of the same experiments. 

    Example:
        loader = DataLoader()
        exp_df = loader.get_default_exp_df()
    """

    def __init__(self):
        self.mean_exp_sets_df = pd.read_csv("./data/mean_exp_sets.csv")

        self.t1_size_grouping_tuple = ['machine', 'workload', 'queue_size', 'thread_count', 'iat_scale', 't1_size']

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


    def get_tier2_performance_df(self):
        """Computes the performance improvement from adding a tier-2 cache. 

        Returns:
            A DataFrame containing MT caches and its performance relative to its corresponding 
            ST cache (same tier-1 size). 
        """
        df_entry_list = []
        for grouping_tuple, df in self.get_default_exp_df().groupby(self.t1_size_grouping_tuple):
            st_df, mt_df = df[df['t2_size']==0], df[df['t2_size']>0]

            if len(st_df) == 0:
                continue 

            st_row = st_df.iloc[0]
            st_bandwidth = st_row['bandwidth_byte/s']
 
            for _, mt_row in mt_df.iterrows():
                # comparing each MT cache to its corresponding ST cache 
                t2_size = mt_row['t2_size']
                mt_bandwidth = mt_row['bandwidth_byte/s']
                percent_diff = 100 * (mt_bandwidth - st_bandwidth)/st_bandwidth

                df_entry_list.append({
                    'machine': grouping_tuple[0],
                    'workload': grouping_tuple[1],
                    't1_size': grouping_tuple[5]/1e3, 
                    't2_size': t2_size/1e3, 
                    't1_hit_rate': mt_row['t1HitRate'],
                    't2_hit_rate': mt_row['t2HitRate'],
                    'percent_diff': percent_diff,
                    't2_t1_size_ratio': t2_size/grouping_tuple[5]
                })
        
        return pd.DataFrame(df_entry_list)

