from ctgan import CTGAN
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from data_modeling import plot_learning_curve
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import metrics



# Load your dataset from the CSV file
real_data = pd.read_csv("CTGAN_DATA.csv")
real_data.drop(columns=['Unnamed: 0'], inplace=True)

unusedX, X_test, unusedY, y_test =  train_test_split(real_data[real_data.columns[real_data.columns != 'Target_Graduate']],
        real_data['Target_Graduate'], test_size=0.25, random_state=1)

# Want to make a dataset from X_test and y_test
# This is to be used by CTGAN to generate synthetic data
data_without_test = pd.concat([unusedX, unusedY], axis=1)

discrete_columns = [
    "Application mode",
    "Application order",
    "Course",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Target_Graduate"
]


# Fit the CTGAN model to your data
ctgan = CTGAN(verbose = True, epochs=1000, batch_size=300, generator_dim=(256, 256, 256), discriminator_dim=(256, 256, 256))
# ctgan.fit(real_data, discrete_columns)

ctgan = CTGAN().load('CTGAN_MODEL.pkl')

# Generate synthetic data
num_samples = len(real_data) 
synthetic_data = ctgan.sample(num_samples)

# Save the synthetic data to a new CSV file
# synthetic_data.to_csv('SYNTHETIC_CTGAN_DATA.csv', index=False)

# ctgan.save('CTGAN_MODEL.pkl')

def training_loss():
    # Get the loss values from the ctgan training as a pd.dataframe with columns=['Epoch', 'Generator Loss', 'Distriminator Loss']
    loss_values = ctgan.loss_values

    # Plot the loss values
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values['Generator Loss'], label="Generator loss")
    plt.plot(loss_values['Discriminator Loss'], label="Discriminator loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# training_loss()
    
gan_data = pd.read_csv("SYNTHETIC_CTGAN_DATA.csv")  

# Merge real and generated data
data = pd.concat([data_without_test, gan_data])

# Get X_train and y_train from data
X_train = data[data.columns[data.columns != 'Target_Graduate']]
y_train = data['Target_Graduate']


# X_train, X_test, y_train, y_test =  train_test_split(data[data.columns[data.columns != 'Target_Graduate']],
#         data['Target_Graduate'], test_size=0.25, random_state=1)
# X_train, unusedx, y_train, unusedy =  train_test_split(data[data.columns[data.columns != 'Target_Graduate']],
        # data['Target_Graduate'], test_size=0.25, random_state=1)
# unusedX, X_test, unusedY, y_test =  train_test_split(real_data[real_data.columns[real_data.columns != 'Target_Graduate']],
#         real_data['Target_Graduate'], test_size=0.25, random_state=1)


        # RANDOM FOREST
def random_forest( X_train, y_train, X_test, y_test, tune = False, learning_curve = False):
    rf_classifier = RandomForestClassifier(max_depth=17, max_features=3, min_samples_leaf=2, n_estimators=300, random_state=42)

    rf_classifier.fit(X_train, y_train)
    # prediction
    y_pred = rf_classifier.predict(X_test)

    if tune:
        param_grid = {'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 9, 13, 17], 
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': [3, 5, 7, 9, 11, 13]}
        
        # Create a based model
        rf = RandomForestRegressor()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)

        # Fit the random search object to the data
        grid_search.fit(X_train, y_train)
        # Create a variable for the best model
        best_rf = grid_search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  grid_search.best_params_)
        print('Best model:', best_rf)
    
    if learning_curve:
        cv = 5
        title = "Learning Curves (Random Forest)"
        plot_learning_curve(rf_classifier, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        plt.show()


    metrics.print_metrics("Random Forest Classifier", y_test, y_pred)

    #This code is based on the svm code found at https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/
def svm(X_train, Y_train, X_test, Y_test, tune = False, learning_curve = False):
    # # Split the data into training and test sets
    # training_set, test_set = train_test_split(real_data, test_size=0.25, random_state=1)

    #standardize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Hyperparameter tuning, from https://www.kaggle.com/code/faressayah/support-vector-machine-pca-tutorial-for-beginner

    print("\n---------")
    print("\nHyperparameter tuning for SVM:\n")

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
        'gamma': [10, 1, 0.5, 0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['rbf', 'poly', 'linear', 'sigmoid']} if tune else {'C': [100], 
                'gamma': [0.001], 
                'kernel': ['rbf']} 


    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)

    best_params = grid.best_params_
    print(f"Best params: {best_params}")

    # Encode the class labels
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)

    # Train the SVM classifier
    classifier = SVC(**best_params)
    classifier.fit(X_train, Y_train)

    # Predict on the test set
    Y_pred = classifier.predict(X_test)
    # test_set["Predictions"] = Y_pred
        
    metrics("SVM", Y_test, Y_pred)

    if learning_curve:
        cv = 5
        title = "Learning Curves (SVM)"
        plot_learning_curve(classifier, title, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        plt.show()
    

random_forest(X_train, y_train, X_test, y_test, tune = False, learning_curve = False)
# svm(X_train, y_train, X_test, y_test, tune = False)


# print(data.shape, synthetic_data.shape)

def corr():

    # compute the correlation matrix
    corr = synthetic_data.corr()

    # plot the heatmap
    f = plt.figure(figsize=(15, 15))
    sns.heatmap(data.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Synthetic Data")
    plt.show()

    original_data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")

    corr = original_data.corr()

    f = plt.figure(figsize=(15, 15))
    sns.heatmap(original_data.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Original Data")
    plt.show()


    # show summary statistics SYNTHETIC DATA
    summary = synthetic_data.describe()
    print(summary)

print("SYNTHETIC DATA")
print(gan_data.describe())
print("\nREAL DATA")
print(real_data.describe())