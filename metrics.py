from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

def print_metrics(modelStr, Y_test=None, Y_pred=None, display_conf_matrix=False):

    print(f"\n------- Metrics for {modelStr} -------\n")

    # Define metrics
    class_report = classification_report(Y_test, Y_pred)
    acc_score = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(list(Y_test), Y_pred)
    

    # Print metrics
    print(class_report)
    print(f"Accuracy for {modelStr}: {acc_score}")
    print(f"F1-Score for {modelStr}: {f1}")

    print(f"Confusion matrix for {modelStr}: \n{conf_matrix}\n")

    if display_conf_matrix:

        # Display metrics
        disp = ConfusionMatrixDisplay(conf_matrix)
        disp.plot()
        plt.show()

def print_avg_metrics(modelStr, preds: list[list], y_test: list):
    '''Assumes same test set for every prediction'''

    print(f"\n------- Average Metrics for {modelStr} -------\n")

    avg_acc = 0
    avg_f1 = 0
    num_runs = 0
    for pred in preds:
        avg_acc += accuracy_score(y_test, pred)
        avg_f1 += f1_score(y_test, pred)
        num_runs += 1

    avg_acc = avg_acc/num_runs
    avg_f1 = avg_f1/num_runs


    print(f"Average accuracy for {modelStr} after {num_runs} run{'s'*min(num_runs-1,1)}: {avg_acc}")
    print(f"Average F1-score for {modelStr} after {num_runs} run{'s'*min(num_runs-1,1)}: {avg_f1}")