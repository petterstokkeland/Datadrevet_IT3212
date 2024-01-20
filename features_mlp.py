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
from MLP import MLP
from data_preprocessing import Preprocessing
from sklearn.metrics import accuracy_score, f1_score

#NB! Executes WITH feature extraction (and selection)
def test_optimal_num_features(max_features=35, max_iterations=3000, runs=5, cv_folds=5, save_to_file=True):

    feature_results = []

    # For every possible number of features
    for num_features in range(1, max_features+1):

        print(f"\n Testing number of features: {num_features}\n")

        # Generate dataset with right number of features
        Preprocessing().gen_mlp_subset(dataset_size=0.1, num_features=num_features, filename="tmp_MLP")

        # Generate model instance using these features
        model = MLP(dataset_file="tmp_MLP")
        preds = model.mlp(max_iterations, runs=3, verbose=False)

        avg_acc = 0
        avg_f1 = 0
        num_runs = 0
        for pred in preds:
            avg_acc += accuracy_score(model.y_test, pred)
            avg_f1 += f1_score(model.y_test, pred)
            num_runs += 1
        avg_acc = avg_acc/num_runs
        avg_f1 = avg_f1/num_runs

        acc_score = accuracy_score(model.y_test, pred)
        f1 = f1_score(model.y_test, pred)

        feature_results.append((num_features, acc_score, f1))

        if save_to_file:
            save_num_features(feature_results)

    return feature_results

def save_num_features(feature_results, save_to_file="MLP_num_features_with_results_v2.csv"):
    df = pd.DataFrame(feature_results, columns=['Number of Features', 'Accuracy', 'F1 Score'])

    # Save the DataFrame to a CSV file
    df.to_csv(save_to_file, index=False)

def plot_num_features(feature_results):

    # Extract data for plotting
    num_features = [result[0] for result in feature_results]
    acc_scores = [result[1] for result in feature_results]
    f1_scores = [result[2] for result in feature_results]

    # Create subplots for acc and f1
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(num_features, acc_scores, marker='o', linestyle='-', color="r")
    plt.title('Accuracy')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')

    plt.xticks(num_features)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(num_features, f1_scores, marker='o', linestyle='-', color="b")
    plt.title('F1 Score')
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')
    
    plt.xticks(num_features)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_num_features():

    df = pd.read_csv("MLP_num_features_with_result.csv")  

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df["Number of Features"], df["Accuracy"], label="Accuracy", marker='o')
    plt.plot(df["Number of Features"], df["F1 Score"], label="F1 Score", marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 Score vs Number of Features")
    plt.xticks(df["Number of Features"])  # Only show integers on the x-axis
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the plot function

feature_res = test_optimal_num_features()
# save_num_features(feature_res)
# plot_num_features(feature_res)
# plot_num_features()

