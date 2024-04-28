import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import Module  # Import your model class
from SampleGenerator import generator

# Perform prediction
with torch.no_grad():
    predictions = []
    correct_answers = 0
    count = 0
    # we will just predict this 100 times and see what the accuracy percentage is
    while count < 100:

        generator.generate_real_sample(1)
        real_sample = pd.read_csv('./real_data.csv')
        scaler = StandardScaler()

        new_scaled_X = scaler.fit_transform(real_sample)

        # Convert data to PyTorch tensor
        new_tensor = torch.FloatTensor(new_scaled_X)

        # Instantiate the model (assuming input_size is known)
        input_size = new_scaled_X.shape[1]
        model = Module.PlantDiseaseModel(input_size)

        # Load the trained model weights
        model.load_state_dict(torch.load('./plant_disease_model.pth'))

        # Set the model to evaluation mode
        model.eval()

        prediction = model(new_tensor)
        predicted_disease_index = torch.argmax(prediction).item()
        predicted_disease_name = Module.y_train.unique()[predicted_disease_index]

        predictions.append({predicted_disease_name,generator.real_disease})

        if predicted_disease_name != generator.real_disease:
            predictions.append(False)
        else:
            correct_answers += 1
        count += 1
        print(f"prediction: {predicted_disease_name}", f"real value: {generator.real_disease}")

    # now do percentage
    print(f"% correct is: {(correct_answers/100)*100}")
