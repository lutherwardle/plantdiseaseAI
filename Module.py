import torch
from SampleGenerator import SampleGenerator
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import pandas as pd


# Define the PyTorch model
class PlantDiseaseModel(nn.Module):
    def __init__(self, input_size):
        super(PlantDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize SoilTestGenerator and generate sample data
generator = SampleGenerator()
sample_tensors = SampleGenerator.generate_sample_tensors(1000)
tensor_data = pd.read_csv('./sample_data.csv')


# Split the data into train and test sets
X = tensor_data.drop("disease_name", axis=1)
y = tensor_data["disease_name"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train_array = np.array([list(sample.values()) for sample in X_train])
# X_test_array = np.array([list(sample.values()) for sample in X_test])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_np = np.array(y)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_np[y_train.index].reshape(-1, 1))
y_test_tensor = torch.FloatTensor(y_np[y_test.index].reshape(-1, 1))

# Instantiate the model and define loss function and optimizer
input_size = X_train_tensor.shape[1]
model = PlantDiseaseModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
    # You can perform additional evaluation metrics if needed, such as RMSE
    rmse = mean_squared_error(y_test_tensor.numpy(), y_pred.numpy(), squared=False)
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

torch.save(model.state_dict(), 'plant_disease_model.pth')
print("Model weights saved successfully.")