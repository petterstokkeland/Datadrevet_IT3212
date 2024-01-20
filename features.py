import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


data = pd.read_csv('graduation_dataset_preprocessed.csv')


""" 
feature engineering: creating two new features
have to do some extra preprocessing on the new features
"""
# approved units rate for 1st and 2nd semester
data['units_approved_rate_1st'] = data['Curricular units 1st sem (approved)']/data['Curricular units 1st sem (enrolled)']
data['units_approved_rate_2nd'] = data['Curricular units 2nd sem (approved)']/data['Curricular units 2nd sem (enrolled)']
# replace NaN with 0, get NaN from new rate features
data.fillna(0, inplace=True)
# check for NaN
# print(nan_in_df = df.isnull().sum().any())

# train/test split of data
X_train, X_test, y_train, y_test = train_test_split(data[data.columns[data.columns != 'Target_Graduate']],data['Target_Graduate'], test_size=0.25, random_state=1)

def heatmap(data):
        f = plt.figure(figsize=(15, 15))
        sns.heatmap(data.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

"""
feature extraction
"""
# filtering method for feature selection, mutual information
def feature_extraction(data, plot):
    x_data = data.drop(['Target_Graduate'], axis=1)
    y_data = data['Target_Graduate']
    mutual_info = mutual_info_classif(x_data, y_data)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_data.columns
    if plot:
        mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 5))
        plt.show()
    return mutual_info

"""
feature selection: filter method
"""
# method for selecting n best features.
def select_n_best(n, data):
    x = data.drop(['Target_Graduate'], axis=1)
    y = data['Target_Graduate']
    # using score function mutual information to capture complex relations
    sel_k_best = SelectKBest(k=n, score_func=mutual_info_classif)
    features = sel_k_best.fit_transform(x,y)
    # print(dict(zip(x.columns, sel_k_best.scores_)))
    return [x.columns[feature] for feature in sel_k_best.get_support(indices=True)]

"""
data transformation using t-sne and pca
"""
# t-sne, can capture complex non-linear relations TODO: fix
#T-SNE
#T-SNE
def t_sne(data, n_component=2, random_s=42):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Initialize t-SNE
    tsne = TSNE(n_components=n_component, random_state=random_s)
    # Fit and transform the data
    tsne_result = tsne.fit_transform(scaled_data)
    # Create a scatter plot of the t-SNE results
    return tsne_result

def plot_t_sne( tsne):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=data['Target_Graduate'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
    plt.colorbar(label='Target_Graduate')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()


# PCA. TODO: fit X_train and X_test and see effects
def pca(data, plot=False, n_components=2):
    # Standardize the data
    x_data = data.drop('Target_Graduate', axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(x_data)
    # Initialize PCA with the number of components you want to retain
    # n_components = 2  # You can adjust this based on your needs
    pca = PCA(n_components=n_components)
    # Fit PCA to the standardized data
    pca_result = pca.fit_transform(scaled_data)
    # Create a scatter plot of the PCA results
    if plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
        plt.colorbar()
        plt.title('PCA Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # plot explained variance, aim for 90%
        pca_new = PCA().fit(scaled_data)
        plt.figure(figsize=(10,7))
        plt.plot(np.cumsum(pca_new.explained_variance_ratio_), color='k', lw=2)
        plt.xlabel('Number of components')
        plt.ylabel('Total explained variance')
        plt.xlim(0, 29)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.axvline(21, c='b')
        plt.axhline(0.91, c='r')
        plt.show();
    # return pca_result


# tune parameters for rf with pca
def hyperparameter_tuning_pca_rf(X_train, X_test, y_train, y_test):
    # Define a pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Scale the data
        ('pca', PCA(n_components=10)),      # Step 2: Apply PCA
        ('rf', RandomForestClassifier())    # Step 3: Train a Random Forest classifier
    ])

    # Using the pipeline to fit and predict
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    # Define parameter grid
    param_grid = {
        'pca__n_components': [2, 5, 10, 25], 
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20]
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(pipe, param_grid, cv=5)

    # Fit
    grid_search.fit(X_train, y_train)

    # Best parameters and estimator
    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_

"""
feature selection: wrapper methods

- probably not feasible as it takes too much computational power
- have not finished the methods
"""


"""
feature selection: embedded methods
should be done together with training
"""

# use random forest for selecting features
def rf_sel(X, y):
    # gini criterion for feature importance
    rfc = RandomForestClassifier(random_state=0, criterion='gini')

    # rfc.fit(X, y)
    # # Print the name and gini importance of each feature
    # for feature in zip(X.columns, rfc.feature_importances_): 
    #     print(feature)

    # select features that have importance above 0.02 using random forest
    sfm = SelectFromModel(estimator=rfc, threshold=0.02)
    sfm.fit(X,y)
    # Print the names of the most important features
    print('The most important features based on random forest classifier:')
    for feature_list_index in sfm.get_support(indices=True):
        print('- ' + X.columns[feature_list_index])
    # Transform the dataset
    X_important_train = sfm.transform(X_train)
    X_important_test = sfm.transform(X_test)
    return X_important_train, X_important_test

def select_best():
    best_features = select_n_best(26, data)
    for feature in data.columns:
        if feature not in best_features and feature != 'Target_Graduate':
            data.drop(feature, axis=1, inplace=True)
    
    kolonne = data.pop('Target_Graduate')
    data.insert(len(data.columns), 'Target_Graduate', kolonne)
    
    data.to_csv('graduation_dataset_preprocessed_feature_selected.csv', index=False)


if __name__ == '__main__':

    # heatmap(data)

    # feature_extraction(data, True)

    """
    exploring data transformation
    """

    # using tsne
    # tsne_2 = t_sne(data=data, n_component=2)
    # tsne_3 = t_sne(data=data, n_component=3)
    # plot_t_sne(tsne_2)
    # plot_t_sne(tsne_3)
    # using pca
    # pca2 = pca(data=data, plot=True, n_components=2)
    # pca21 = pca(data=data, plot=True, n_components=21)
    # best_model = hyperparameter_tuning_pca_rf(X_train, X_test, y_train, y_test)

    # 
    # comparing select k best, rf with pca, feature selction with random forest and just the full dataset
    # on random forest to see how it impacts accuracy score
    # 
    """
    sel_columns = select_n_best(17, data)
    print('The most important features based on Select_K_Best:')
    for feature in sel_columns:
        print('- ' + feature)

    x_sel_train, x_sel_test = X_train[[column for column in sel_columns]], X_test[[column for column in sel_columns]]

    X_important_train, X_important_test = rf_sel(X_train,y_train)
    
    # try out optimal hyperparameters from best_model on pca and rf
    # 
    # Standardize the data 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # initiate pca with optimal number of components
    pca = PCA(n_components=25)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    # initate random forest with optimal depth and estimators
    rf = RandomForestClassifier(max_depth=20, n_estimators=200)
    rf.fit(X_train_pca, y_train)
    # Evaluate the model
    accuracy = rf.score(X_test_pca, y_test)
    print(f"Accuracy using PCA with some parameter tuning: {accuracy * 100:.2f}%")

    # Create a random forest classifier
    rfc_full = RandomForestClassifier(random_state=0, criterion='gini')
    # Train the classifier using dataset with full features
    rfc_full.fit(X_train, y_train)
    # Make predictions
    pred_full = rfc_full.predict(X_test)
    # Generate accuracy score
    print('The accuracy of classifier with full features: {:.2f}'.format(accuracy_score(y_test, pred_full)))

    # random forest with feature selection using random forest
    rfc_lim = RandomForestClassifier(random_state=0, criterion='gini')
    # Train the classifier with limited features
    rfc_lim.fit(X_important_train, y_train)
    # Make predictions
    pred_lim = rfc_lim.predict(X_important_test)
    # Generate accuracy score
    print('The accuracy of classifier with features selected with random forest: {:.2f}'.format(accuracy_score(y_test, pred_lim)))

    # random forest with feature selection using select k best
    rfc_sel = RandomForestClassifier(random_state=0, criterion='gini')
    # Train the classifier with limited features
    rfc_sel.fit(x_sel_train, y_train)
    # Make predictions
    pred_sel = rfc_sel.predict(x_sel_test)
    # Generate accuracy score
    print('The accuracy of classifier with features selected with select k best: {:.2f}'.format(accuracy_score(y_test, pred_sel)))
    """
    # Notes:
    
    # - should balance out data when using decision trees, biased to imbalance
    # - bagging and boosting can improve performance, ensemble learning (advanced models?)
    # - may not be neccessary with pca or other data transformation/dimension reduction
    # - can do hyperparameter tuning using GridSearchCV
    # print(feature_extraction(data, True))
    select_best()



