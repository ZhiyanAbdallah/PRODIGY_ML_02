import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load the dataset from Google Drive with error handling
try:
    data = pd.read_csv('/content/drive/My Drive/AmesHousing.csv')  # Path to your dataset
except FileNotFoundError:
    print("Error: Dataset not found. Please check the file path and try again.")
    exit()

# Fill missing values with mean for numeric columns only
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Select features and target
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', 'Total Bsmt SF', '1st Flr SF', 'Year Built']
X = data[features]
y = data['SalePrice']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the neural network model
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Input to hidden layer 1
        self.fc2 = nn.Linear(64, 32)          # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(32, 1)           # Hidden layer 2 to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation after layer 1
        x = torch.relu(self.fc2(x))  # ReLU activation after layer 2
        x = self.fc3(x)              # No activation for the output
        return x

# Instantiate the model, loss function, and optimizer
model = HousePriceModel(input_size=X_train.shape[1])
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the training loss
plt.plot(train_losses)
plt.title("Training Loss over Epochs")
plt.show()

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")

# Function to get user input for prediction
def get_user_input():
    try:
        overall_qual = float(input("Enter Overall Quality (1-10): "))
        gr_liv_area = float(input("Enter Ground Living Area (in square feet): "))
        garage_cars = float(input("Enter number of cars the garage can hold: "))
        garage_area = float(input("Enter Garage Area (in square feet): "))
        total_bsmt_sf = float(input("Enter Total Basement Area (in square feet): "))
        first_flr_sf = float(input("Enter 1st Floor Area (in square feet): "))
        year_built = float(input("Enter Year Built: "))

        return [[overall_qual, gr_liv_area, garage_cars, garage_area, total_bsmt_sf, first_flr_sf, year_built]]
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return None

# Get and process user data
user_data = get_user_input()
if user_data:
    try:
        user_data_scaled = scaler.transform(user_data)
        user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)

        # Make prediction
        model.eval()
        with torch.no_grad():
            user_prediction = model(user_data_tensor)
        print(f"The predicted house price is: ${user_prediction.item():.2f}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
else:
    print("Prediction skipped due to invalid input.")
