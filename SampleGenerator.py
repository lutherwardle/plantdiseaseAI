import csv
import json
import numpy as np


class SampleGenerator:
    real_disease = 0.0
    def __init__(self):
        pass


    """
    the samples themselves are entirely random... we need to change some to make them seem like real data
    for example we will change the samples so that any sample that has over 80% saturation and over 50% macronutrients
    will have its is_diseased attribute set to true and have it's disease name set to redspotfever
    """
    @staticmethod
    def normalize_sample(sample):
        # create a correlation for various diseases (these are obviously fake but we can replace them with real one's later)
        if sample['base_saturation'] > 80.00 and sample['manganese'] > 50.00:
            sample['disease_name'] = 1.0
            return sample

        if sample['pH'] > 30.00 and sample['boron'] > 20.00:
            sample['disease_name'] = 2.0
            return sample

        if sample['potassium'] > 50.00 and sample['calcium'] > 20.00:
            sample['disease_name'] = 3.0
            return sample

        sample['disease_name'] = 0.0
        return sample

    @staticmethod
    def determine_if_estimation_is_correct(sample):
        for item in sample:
            if item['base_saturation'] > 80.00 and item['manganese'] > 50.00:
                return 1.0

            if item['pH'] > 30.00 and item['boron'] > 20.00:
                return 2.0

            if item['potassium'] > 50.00 and item['calcium'] > 20.00:
                return 3.0

        return 0.0

    @staticmethod
    def generate_sample_tensors(number_to_make):
        sample_tensors = []
        for _ in range(number_to_make):
            # Generate random values for each metric
            pH = round(np.random.uniform(4.0, 9.0), 2)
            nitrogen = round(np.random.uniform(0.1, 100.0), 2)
            phosphorus = round(np.random.uniform(0.1, 100.0), 2)
            potassium = round(np.random.uniform(0.1, 100.0), 2)
            calcium = round(np.random.uniform(0.1, 100.0), 2)
            magnesium = round(np.random.uniform(0.1, 100.0), 2)
            sulfur = round(np.random.uniform(0.1, 100.0), 2)
            iron = round(np.random.uniform(0.1, 100.0), 2)
            manganese = round(np.random.uniform(0.1, 100.0), 2)
            zinc = round(np.random.uniform(0.1, 100.0), 2)
            copper = round(np.random.uniform(0.1, 100.0), 2)
            boron = round(np.random.uniform(0.1, 100.0), 2)
            molybdenum = round(np.random.uniform(0.1, 100.0), 2)
            chlorine = round(np.random.uniform(0.1, 100.0), 2)
            organic_matter = round(np.random.uniform(0.1, 100.0), 2)
            cation_exchange_capacity = round(np.random.uniform(5.0, 50.0), 2)
            base_saturation = round(np.random.uniform(40.0, 90.0), 2)
            soil_texture = np.random.choice([0.11, 0.12, 0.13, 0.14])
            disease_name = 0.0

            # Create a dictionary representing the sample tensor
            sample_tensor = {
                "pH": pH,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "calcium": calcium,
                "magnesium": magnesium,
                "sulfur": sulfur,
                "iron": iron,
                "manganese": manganese,
                "zinc": zinc,
                "copper": copper,
                "boron": boron,
                "molybdenum": molybdenum,
                "chlorine": chlorine,
                "organic_matter": organic_matter,
                "cation_exchange_capacity": cation_exchange_capacity,
                "base_saturation": base_saturation,
                "soil_texture": soil_texture,
                "disease_name": disease_name
            }

            filtered_sample = SampleGenerator.normalize_sample(sample_tensor)
            sample_tensors.append(filtered_sample)

        return sample_tensors

    @staticmethod
    def generate_real_sample(number_to_make):
        real_tensors = []
        for _ in range(number_to_make):
            # Generate random values for each metric
            pH = round(np.random.uniform(4.0, 9.0), 2)
            nitrogen = round(np.random.uniform(0.1, 100.0), 2)
            phosphorus = round(np.random.uniform(0.1, 100.0), 2)
            potassium = round(np.random.uniform(0.1, 100.0), 2)
            calcium = round(np.random.uniform(0.1, 100.0), 2)
            magnesium = round(np.random.uniform(0.1, 100.0), 2)
            sulfur = round(np.random.uniform(0.1, 100.0), 2)
            iron = round(np.random.uniform(0.1, 100.0), 2)
            manganese = round(np.random.uniform(0.1, 100.0), 2)
            zinc = round(np.random.uniform(0.1, 100.0), 2)
            copper = round(np.random.uniform(0.1, 100.0), 2)
            boron = round(np.random.uniform(0.1, 100.0), 2)
            molybdenum = round(np.random.uniform(0.1, 100.0), 2)
            chlorine = round(np.random.uniform(0.1, 100.0), 2)
            organic_matter = round(np.random.uniform(0.1, 100.0), 2)
            cation_exchange_capacity = round(np.random.uniform(5.0, 50.0), 2)
            base_saturation = round(np.random.uniform(40.0, 90.0), 2)
            soil_texture = np.random.choice([0.11, 0.12, 0.13, 0.14])

            # Create a dictionary representing the sample tensor
            real_tensor = {
                "pH": pH,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "calcium": calcium,
                "magnesium": magnesium,
                "sulfur": sulfur,
                "iron": iron,
                "manganese": manganese,
                "zinc": zinc,
                "copper": copper,
                "boron": boron,
                "molybdenum": molybdenum,
                "chlorine": chlorine,
                "organic_matter": organic_matter,
                "cation_exchange_capacity": cation_exchange_capacity,
                "base_saturation": base_saturation,
                "soil_texture": soil_texture,
            }

            real_tensors.append(real_tensor)

            csv_real_sample = './real_data.csv'

            # generate and store a "real" sample
            generator.real_disease = generator.determine_if_estimation_is_correct(real_tensors)
            with open(csv_real_sample, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=real_tensors[0].keys())
                writer.writeheader()
                writer.writerows(real_tensors)

        return real_tensors



# Example usage:
generator = SampleGenerator()
sample_tensors = generator.generate_sample_tensors(1000)
real_tensors = generator.generate_real_sample(1)

# Specify the output CSV file name
csv_file = './sample_data.csv'

# Write data to CSV file for samples
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=sample_tensors[0].keys())
    writer.writeheader()
    writer.writerows(sample_tensors)

print(f"CSV file '{csv_file}' has been created.")