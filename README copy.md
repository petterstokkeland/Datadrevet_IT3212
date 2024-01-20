# Datadrevet-gruppe-12

## Student Enrollment and Academic Performance Dataset

### Overview

This GitHub repository contains a comprehensive dataset providing detailed information about students' enrollment in various courses, their personal characteristics, academic performance, and economic indicators. This dataset is a valuable resource for data preprocessing and modeling techniques using Python.

### Dataset Description

The dataset encompasses a wide range of fields, including:

- **Personal Characteristics**: Details about students' personal attributes, such as marital status, age at enrollment, and application mode, offering insights into the diversity of students and their preferred application methods.

- **Educational Backgrounds**: Information about students' previous qualifications and the qualifications of their parents, which could potentially influence their academic journey.

- **Attendance Preferences**: Data on students' attendance preferences, nationality, and special needs, allowing for a more nuanced understanding of the student population.

- **Curricular Units**: Information on the curricular units in the first and second semesters, covering credit details, evaluations, approvals, and grades.

- **Economic Indicators**: Key economic indicators such as the unemployment rate, inflation rate, and GDP, which may offer insights into any correlations between economic conditions and academic outcomes.

- **Target Variable**: The 'Target' column is the focal point for analysis, indicating whether a student has graduated after normal time or not. 

### Repository Contents

This repository contains the following Python classes and files:

- **.gitignore**: A file specifying which files and directories to exclude from version control.

- **LICENSE**: The license information for using the dataset and code in this repository.

- **README.md**: This readme file, providing an overview of the dataset and instructions for using the repository.

- **data_modeling.py**: A Python file that contains a class that showcases various data modeling techniques using Python. It contains two basic modeling techniques, Support Vector Machine (SVM), and Random Forest Classifier (RFC). It runs on the "graduation_dataset_preprocessed_feature_selected.csv", that has been preprocessed, and feature selected and extracted. 

- **data_preprocessing.py**: Another Python file containing a class that demonstrates Python methods for data preprocessing on the dataset. It one-hot encodes the target column. It also contains various methods for visualizing the data. It truns on the "graduation_dataset.csv", that we got from the source. 

- **features.py**: A Python file that performed feature selection and extraction. It runs on the "graduation_dataset_preprocessed.csv", that has been preprocessed. 

- **graduation_dataset.csv**: The dataset file in CSV format containing all the student enrollment and academic performance data. This is the dataset we got from the source. 

- **graduation_dataset_preprocessed.csv**: An additional dataset file that has undergone preprocessing.

- **graduation_dataset_preprocessed_feature_selected.csv**: An additional dataset file that has undergone both preprocessing and feature selection.

### Usage

To get started with using this dataset and the Python classes for data preprocessing and modeling, follow these steps:

1. Clone or download this repository to your local machine.

2. Open the Python classes (data_preprocessing.py, data_modeling.py, and features.py).

3. Follow the instructions and code within the classes and files to explore, preprocess, and model the data. In data_modeling.py, you can uncomment the methods you want to run in the main method at the bottom. 

4. You can use the provided dataset (graduation_dataset.csv) or replace it with your own data, making sure it follows a similar structure.

### Citation

If you use this dataset or find the provided code and techniques useful for your research or analysis, please consider citing the repository to acknowledge the source.

### Acknowledgments

We would like to express our gratitude to the contributors and organizations that made this dataset available for research and analysis.

For any questions, issues, or suggestions related to this repository, please feel free to create an issue or contact the repository owner.

Happy data preprocessing and modeling! 
