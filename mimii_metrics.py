import numpy as np
from sklearn import metrics

class AUCMetrics:
    def __init__(self, folder_name):
        self.target_img = []
        self.y_true = []
        self.output_img = []

    def update(self, data, tensor_values):
        self.target_img.extend(data['target_img'])
        self.y_true.extend(data['y_true'])
        self.output_img.extend(tensor_values['output_img'])

    def get_worker_results(self):
        return {'target_img': self.target_img, 
                'output_img': self.output_img, 
                'y_true': self.y_true}
    

    def reduce_all_worker_results(self, worker_results_list):
        target_img = []
        output_img = []
        y_true = []

        for worker in worker_results_list:
            target_img.extend(worker['target_img'])
            output_img.extend(worker['output_img'])
            y_true.extend(worker['y_true'])

        target_img = np.array(target_img)
        output_img = np.array(output_img)
        y_true = np.array(y_true)
    
        errors = np.square(target_img - output_img).mean(axis=2)
        y_pred = np.mean(errors,axis=1)

        auc = metrics.roc_auc_score(y_true, y_pred)

        print(f"AUC: {auc:2.3f}")
        return {'AUC':auc}
