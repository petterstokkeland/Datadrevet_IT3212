import numpy as np
import pandas as pd
import random
from ast import If
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split

class Preprocessing:

    def __init__(self) -> None:
        # Load your dataset
        self.data = pd.read_csv("graduation_dataset.csv")  # Replace with your dataset file path
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target']],self.data['Target'], test_size=0.25, random_state=1)

    def one_hot_encoding(self):
        self.data = pd.get_dummies(self.data, columns=['Target']) 
        self.data = self.data.astype(int) #endrer true og false til 1 og 0
        self.data.drop('Target_Enrolled', inplace=True, axis=1)
        self.data.drop('Target_Dropout', inplace=True, axis=1)
        
    def standardizeMinMax(self):
        scaler = MinMaxScaler()
        self.data.iloc[:, :] = scaler.fit_transform(self.data.iloc[:, :])    
        
    def describe(self):
        print(self.data.head())
        print(self.data.describe())

    def heatmap(self):
        # Removing the last column
        self.data = self.data.iloc[:, :-1]
        f = plt.figure(figsize=(15, 15))
        sns.heatmap(self.data.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()


    # BEST FEATURE SELECTION
    def best_feature_selection(self, plot=False):
        mutual_info = mutual_info_classif(self.X_train, self.y_train)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = self.X_train.columns
        if plot:
            mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 5))
            plt.show()
        return mutual_info


    #T-SNE
    def t_sne(self):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        # Initialize t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        # Fit and transform the data
        tsne_result = tsne.fit_transform(scaled_data)
        # Create a scatter plot of the t-SNE results

        return tsne_result

    def plot_t_sne(self, tsne):
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne[:, 0], tsne[:, 1], c=self.data['Target'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
        plt.colorbar(label='Target')
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    # PCA
    def pca(self, plot=False, n_components=2):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        # Initialize PCA with the number of components you want to retain
        # n_components = 2  # You can adjust this based on your needs
        pca = PCA(n_components=n_components)
        # Fit PCA to the standardized data
        pca_result = pca.fit_transform(scaled_data)
        # Create a scatter plot of the PCA results
        if plot:
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.data['Target'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
            plt.colorbar(label='Target')
            plt.title('PCA Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()
        
        return pca_result
    
    def generate_features(self):
        # approved units rate for 1st and 2nd semester
        self.data['units_approved_rate_1st'] = self.data['Curricular units 1st sem (approved)']/self.data['Curricular units 1st sem (enrolled)']
        self.data['units_approved_rate_2nd'] = self.data['Curricular units 2nd sem (approved)']/self.data['Curricular units 2nd sem (enrolled)']
        # replace NaN with 0, get NaN from new rate features
        self.data.fillna(0, inplace=True)

        # Move target to last column
        column = self.data.pop('Target_Graduate')
        self.data.insert(len(self.data.columns), 'Target_Graduate', column)

    # method for selecting n best features.
    def find_n_best(self, n):
        x = self.data.drop(['Target_Graduate'], axis=1)
        y = self.data['Target_Graduate']
        # using score function mutual information to capture complex relations
        sel_k_best = SelectKBest(k=n, score_func=mutual_info_classif)
        features = sel_k_best.fit_transform(x,y)
        # print(dict(zip(x.columns, sel_k_best.scores_)))
        return [x.columns[feature] for feature in sel_k_best.get_support(indices=True)]

    def select_best(self, num_features):
        best_features = self.find_n_best(num_features)
        for feature in self.data.columns:
            if feature not in best_features and feature != 'Target_Graduate':
                self.data.drop(feature, axis=1, inplace=True)

    def reduce_dataset(self, dataset_size=1):
        """
        Reduces the size of the dataset by randomly selecting a subset of rows based on the specified dataset size.

        Parameters:
        - dataset_size (float, optional): A float between 0 (exclusive) and 1 (inclusive) specifying the fraction of the dataset to retain. Defaults to 1, which means no reduction.

        Raises:
        - ValueError: If the dataset_size is not between 0 (exclusive) and 1 (inclusive).

        Note:
        - Recompute training and test sets
        """
        if not 0 < dataset_size <= 1:
            raise ValueError("Column percentage must be between 0 and 1")
        
        #Choose random subset of columns
        self.data = self.data.sample(frac=dataset_size)

        #Recompute training/test sets
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target']],self.data['Target'], test_size=0.25, random_state=1)

    
    def generate_dataset_ensemble(self):
        self.one_hot_encoding()
        self.standardizeMinMax()
        self.data.to_csv("graduation_dataset_preprocessed.csv")

    def generate_dataset_mlp(self, num_features, feature_extract=True, feature_select=True):
        self.one_hot_encoding()
        if feature_extract and feature_select:
            self.generate_features()
            self.select_best(num_features)
            self.data.to_csv("MLP_graduation_dataset_preprocessed_feature_extracted_selected.csv")
        elif feature_extract:
            self.generate_features()
            self.data.to_csv("MLP_graduation_dataset_preprocessed_feature_extracted.csv")
        elif feature_select:
            self.select_best(num_features)
            self.data.to_csv("MLP_graduation_dataset_preprocessed_feature_selected.csv")
        else:
            self.data.to_csv("MLP_graduation_dataset_preprocessed.csv")

    def gen_mlp_subset(self, dataset_size=1, num_features=35, filename="MLP_feature_selection_testing"):
        self.reduce_dataset(dataset_size)
        self.one_hot_encoding()
        self.generate_features()
        self.select_best(num_features)
        self.data.to_csv(filename)


if __name__ == "__main__":
    print("Happy data preprocessing and modeling!")
    preprocessing = Preprocessing()
    preprocessing.generate_dataset_mlp(13)
