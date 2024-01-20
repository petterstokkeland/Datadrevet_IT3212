

import numpy as np
import pandas as pd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from features import select_best
import time
import ensemble
import metrics

class MLP:

    def __init__(self, train_test_sets=None, dataset_with_feature_extraction=True, dataset_file=None) -> None:
        # Load your dataset

        if dataset_file is not None:
            self.data = pd.read_csv(dataset_file)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target_Graduate']],
                                                                                    self.data['Target_Graduate'], test_size=0.25, random_state=1)
            return

        if train_test_sets is None:
            if dataset_with_feature_extraction:
                self.data = pd.read_csv("MLP_graduation_dataset_preprocessed_feature_extracted_selected.csv")
            else:
                self.data = pd.read_csv("MLP_graduation_dataset_preprocessed_feature_selected.csv")
            # Split the data into test and train
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target_Graduate']],
                                                                                    self.data['Target_Graduate'], test_size=0.25, random_state=1)
        else:
            # Use inputed traing and test-sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_sets

        


    def mlp_single(self, maxIterations,tune=False, verbose=True):
        #standardize data
        #from the documentation https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(self.X_train)  
        self.X_train = scaler.transform(self.X_train)  
        # apply same transformation to test data
        self.X_test = scaler.transform(self.X_test)

        #hyperparameter tuning
        if tune:
            parameter_space = {
                'hidden_layer_sizes': [(16),(16,8),(16,16),(16,8,8),(16,16,8),(16,16,16)], #trying diffrent number of layers and difrrent neuron count
                'alpha': 10.0 ** -np.arange(1, 7),
                'activation': ['identity', 'logistic', 'tanh', 'relu']

            }
            grid = GridSearchCV(MLPClassifier(max_iter=maxIterations), parameter_space, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)

            best_params = grid.best_params_
            if verbose:
                print(f"Best params: {best_params}")
        else:
            best_params={'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 100, 50)}
       
        # Train the SVM classifier
        mlp = MLPClassifier(max_iter=maxIterations, **best_params)
        mlp.fit(self.X_train, self.y_train)

        # #predict the test set
        predictions = mlp.predict(self.X_test)
        # #print the metrics
        if verbose:
            metrics.print_metrics("MLP", self.y_test, predictions)
        return predictions

    def mlp(self, maxIterations,tune=True, runs=1, verbose=True):

        pred_avg = []

        for _ in range(runs):
            pred= self.mlp_single(maxIterations,tune, verbose)
            pred_avg.append(pred)
        
        return pred_avg

    def plottingFeatures(self):
        print("Happy data preprocessing and modeling!")
        y=[]
        x=list(range(1,35))
        max_iterations=5000
        number_of_runs=3
        best_parameters=[{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 1e-06, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100,)},{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,)},{'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,)},{'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'identity', 'alpha': 1e-06, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 1e-06, 'hidden_layer_sizes': (100,)},{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 0.0001, 'hidden_layer_sizes': (100,)},{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)}]


        for i in range(1,35):
            select_best(i)
            datamodeling = self.data_modeling()
            runs=[]
            #getting avg accuracy
            runs=[datamodeling.mpl(datamodeling.X_train, datamodeling.y_train, datamodeling.X_test, datamodeling.y_test,max_iterations,True,best_parameters[i-1]) for i in range(number_of_runs)]
            print(f"Average F1 for MLP after {number_of_runs} run{'s'*min(number_of_runs-1,1)}: {sum(runs)/number_of_runs}")
            y+=[sum(runs)/number_of_runs]
            print(f"Iteration {i}/34")
        print(y)

        plt.plot(x,y)
        plt.xlabel("Number of features")
        plt.ylabel("F1 score")
        plt.show()

    def plotAccuracyOfHyperParamaterTuningRuns(self):
        twoAvg=[0.8264014466546112, 0.8313743218806511, 0.8300180831826401, 0.8282097649186257, 0.8250452079566004, 0.8345388788426763, 0.8471971066907775, 0.846745027124774, 0.852622061482821, 0.8435804701627487, 0.8594032549728752, 0.8612115732368897, 0.8666365280289331, 0.8648282097649187, 0.8648282097649187, 0.8625678119349005, 0.8652802893309222, 0.8557866184448464, 0.8607594936708861, 0.8612115732368897, 0.8639240506329113, 0.8512658227848101, 0.8594032549728752, 0.8612115732368897, 0.860759493670886, 0.8589511754068716, 0.860759493670886, 0.8557866184448463, 0.8575949367088608, 0.8575949367088608, 0.8603074141048825, 0.8566907775768535, 0.8566907775768535, 0.8521699819168174]
        oneAvg=[0.8264014466546112, 0.8309222423146474, 0.8318264014466547, 0.8300180831826401, 0.825497287522604, 0.8426763110307414, 0.8390596745027125,  0.8417721518987342,0.8544303797468354,0.8589511754068716, 0.8553345388788427, 0.8535262206148282, 0.8688969258589512, 0.8625678119349005, 0.8526220614828209, 0.8607594936708861, 0.8679927667269439, 0.8634719710669078, 0.8607594936708861, 0.8625678119349005, 0.8571428571428571, 0.8598553345388789, 0.8643761301989150, 0.8598553345388789, 0.8589511754068716, 0.859855334538879, 0.8598553345388789, 0.857142857142857, 0.8580470162748643, 0.8580470162748643, 0.8580470162748643, 0.8481012658227848, 0.8562386998010185, 0.8616636528028933]

        y=[(2*twoAvg[i]+oneAvg[i])/3 for i in range(len(oneAvg))] 
        x=range(1,len(y)+1)

        standarddeviation= []
        plt.plot(x,y)
        #x axis show every number
        plt.xticks(x)

        #plot error between twoAvg and oneAvg
        #plt.plot(x,twoAvg,'r--')
        #plt.plot(x,oneAvg,'y--')

        #add ledgend
        plt.legend(['Average Accuracy'])
        #plt.legend(['Average Accuracy overall','Accuracy of 2 runs','Accuracy of 1 run'])



        #mark y ticks every 0.01 and the max and min values aswell
        plt.yticks([i/100 for i in range(82,87)]+[0.825204,0.8672])
        #plot the points in blue
        plt.plot(x,y,'bo')

        #add grid, with x=1 spacing, in grey, alternate dashes and lines
        plt.grid(True, which='major', linestyle='--', color='grey')
        plt.title('Average Accuracy of 3 runs vs. Number of Features')
        plt.xlabel('Number of features')
        plt.ylabel('Average Accuracy')
        plt.show()         


def compare(model_predictions):
    """
    Compare results of different models.
    Must be run on same test-set.

    model_predictions should be on list format.\n 
    Eg: [modelpreds1, modelpreds2, ...]
    """

    test_length = len(model_predictions[0])

    disagreements = 0

    for i in range(test_length):
    # Compare element on index for all models
        prediction_at_index = [model[i] for model in model_predictions]

        print()
        # Check if there are differences among the model predictions
        if len(set(prediction_at_index)) > 1:
            disagreements += 1

    print('Total number of disagreements is ' + str(disagreements))
    print('The length of the test set is ' + str(test_length))
    print('Fraction of disagreements is ' + str(round(disagreements/test_length, 3)))
    return disagreements

def getTestTrainSets():
    data = pd.read_csv("MLP_graduation_dataset_preprocessed_feature_selected.csv")  

    # Split the data into test and train
    return train_test_split(data[data.columns[data.columns != 'Target_Graduate']],
    data['Target_Graduate'], test_size=0.25, random_state=1)




if __name__ == '__main__':

    max_iterations=3000
    number_of_runs=3

    #start timer
    start_time = time.time()
    
    datamodeling = MLP()
    mlp_preds = datamodeling.mlp(max_iterations, runs=number_of_runs)

    print("--- %s seconds ---" % (time.time() - start_time))
    # metrics.print_avg_metrics("MLP", mlp_preds, y_test)
    