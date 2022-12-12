import pathlib
import numpy as np 
import pandas as pd  
from lazypredict.Supervised import LazyClassifier, LazyRegressor

class MLExperiment:
    def __init__(self):
        self.data_path = pathlib.Path("./data/cp_ml_data.csv")
        self.data = pd.read_csv(self.data_path)

        # test sets to evaluate 
        self.test_workload_set_list = [
            ['w105'],
            ['w105', 'w81'],
            ['w105', 'w82'],
            ['w105', 'w97'],
            ['w105', 'w77'],
            ['w105', 'w78'],
            ['w105', 'w47'],
            ['w105', 'w11'],
            ['w105', 'w20'],
            ['w105', 'w68'],
            ['w105', 'w54'],
            ['w105', 'w03'],
            ['w105', 'w13'],
            ['w105', 'w31'],
            ['w105', 'w54', 'w68'],
            ['w105', 'w13', 'w77'],
            ['w105', 'w13', 'w03'],
            ['w105', 'w31', 'w97', 'w47'],
            ['w105', 'w31', 'w78', 'w47'],
            ['w105', 'w47', 'w11', 'w20'],
            ['w105', 'w47', 'w81', 'w81'],
            ['w105', 'w47', 'w81', 'w81', 'w68', 'w20', 'w11'],
            ['w105', 'w47', 'w97', 'w78', 'w77', 'w20', 'w11']
        ]


    def get_test_train_data(self, test_workload_list):
        test = self.data[self.data['workload'].isin(test_workload_list)]
        train = self.data[~self.data['workload'].isin(test_workload_list)]

        train = train.drop(['workload', 't1_hit_rate', 't2_hit_rate', 'miss_rate'], axis=1)
        test = test.drop(['workload', 't1_hit_rate', 't2_hit_rate', 'miss_rate'], axis=1)

        return train.to_numpy(), test.to_numpy()


    def get_max_potential(self, percent_improve_list):
        copy_percentage_improve_list = np.copy(percent_improve_list)
        copy_percentage_improve_list[copy_percentage_improve_list<0]=0
        return copy_percentage_improve_list.sum()


    def eval_classification_models(self, models, predictions, y_test, percent_improve_list):
        model_eval_list = []
        potential_percent_list = []
        max_potential_improve = self.get_max_potential(percent_improve_list)
        for index, model_row in models.iterrows():
            cur_model = model_row.name
            cur_model_pred = predictions[cur_model]
            cur_model_accuracy = model_row['Accuracy']

            # additional model metrics 
            model_potential_loss = 0 
            false_negative_count, negative_count = 0, 0 
            false_positive_count, positive_count = 0, 0 
            for pred_index in range(len(cur_model_pred)):
                cur_pred = cur_model_pred.iloc[pred_index]
                cur_target = y_test[pred_index]

                if cur_pred != cur_target:
                    # misclassification 
                    if cur_target == 0:
                        # we added a tier when we should not have 
                        model_potential_loss += abs(percent_improve_list[pred_index])
                        false_positive_count += 1
                        negative_count += 1
                    else:
                        # we did not add a tier when we should have 
                        model_potential_loss += percent_improve_list[pred_index]
                        false_negative_count += 1
                        positive_count += 1
                else:
                    # correct classification! 
                    if cur_target == 0:
                        # sucessfully avoided adding a tier 
                        negative_count += 1 
                    else:
                        # sucessfully added a tier 
                        positive_count += 1

            potential_loss_ratio = model_potential_loss/max_potential_improve if max_potential_improve > 0 else np.inf 
            false_negative_rate = false_negative_count/positive_count if positive_count > 0 else np.inf 
            false_positive_rate = false_positive_count/negative_count if negative_count > 0 else np.inf 

            model_eval_list.append({
                'model': cur_model,
                'accuracy': cur_model_accuracy,
                'potential_loss_ratio': potential_loss_ratio,
                'false_negative_rate': false_negative_rate,
                'false_positive_rate': false_positive_rate,
                'true_positive_count': positive_count,
                'true_negative_count': negative_count
            })

        df = pd.DataFrame(model_eval_list)
        return df[df['potential_loss_ratio']==df['potential_loss_ratio'].min()].iloc[0]
        

    def run(self):
        # track the result of each test workload set 
        classification_result_list = []
        # iterate through each test workload set 
        for test_workload_set in self.test_workload_set_list:
            train, test = self.get_test_train_data(test_workload_set)
            train_x, test_x = train[:, :-1], test[:, :-1] 
            train_y, test_y = np.where(train[:, -1]>0, 1, 0), np.where(test[:, -1]>0, 1, 0)

            clf = LazyClassifier(predictions=True)
            models, predictions = clf.fit(train_x, test_x, train_y, test_y)

            out = self.eval_classification_models(models, predictions, test_y, test[:, -1])
            classification_result = {
                'test': ','.join([str(_) for _ in test_workload_set]),
                'train_size': len(train_x),
                'test_size': len(test_x)
            }
            for key in out.index:
                classification_result[key] = out[key]
            
            classification_result_list.append(classification_result)
            print(classification_result)
        
        df = pd.DataFrame(classification_result_list)
        print(df)
        df.to_csv('ml_experiment_output.csv', index=False)


if __name__ == "__main__":
    experiment = MLExperiment()
    experiment.run()