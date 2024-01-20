import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
# Load dataset

data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")  # Replace with your dataset file path


# Split data
X = data.drop("Target_Graduate", axis=1).values
y = data["Target_Graduate"].values
# Use SMOTE on training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Split the data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1)
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)
# Define the MLP model
# ... [resten av koden foran er uendret]
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.layer3(x)  # Fjern sigmoid her
        return x
model = MLP(X_train.shape[1])
# Bruk BCEWithLogitsLoss som tapsfunksjon
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# ... [resten av trenings- og evalueringskoden er uendret]
# Tren modellen
epochs = 200
batch_size = 32
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    # Oppdater læringshastigheten ved slutten av hver epoke
    scheduler.step()
# Evaluer modellen
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test, y_pred_class)
    confusion = confusion_matrix(y_test, y_pred_class)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)