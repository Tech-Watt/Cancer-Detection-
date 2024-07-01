# Cancer Prediction Project
## Introduction
This project aims to develop a machine learning model to predict the level of cancer risk based on various patient characteristics and medical history. 
The project uses a dataset of cancer patients with information on their age, gender, environmental factors, medical history, and symptoms.

## Dataset
The dataset used in this project is named "cancer data.csv" and contains the following features:

- `Patient Id`: Unique identifier for each patient
- `Age`: Age of the patient
- `Gender`: Gender of the patient (1 for male, 0 for female)
- `Air Pollution`: Level of air pollution exposure (1-5 scale)
- `Alcohol use`: Level of alcohol consumption (1-5 scale)
- `Dust Allergy`: Level of dust allergy (1-6 scale)
- `OccuPational Hazards`: Level of occupational hazards (1-5 scale)
- `Genetic Risk`: Level of genetic risk (1-5 scale)
- `chronic Lung Disease`: Level of chronic lung disease (1-4 scale)
- `Balanced Diet`: Level of balanced diet (1-5 scale)
- `Obesity`: Level of obesity (1-5 scale)
- `Smoking`: Level of smoking (1-5 scale)
- `Passive Smoker`: Level of passive smoking exposure (1-5 scale)
- `Chest Pain`: Level of chest pain (1-10 scale)
- `Fatigue`: Level of fatigue (1-5 scale)
- `Weight Loss`: Level of weight loss (1-5 scale)
- `Shortness of Breath`: Level of shortness of breath (1-10 scale)
- `Wheezing`: Level of wheezing (1-10 scale)
- `Swallowing Difficulty`: Level of swallowing difficulty (1-6 scale)
- `Clubbing of Finger Nails`: Level of clubbing of finger nails (1-2 scale)
- `Frequent Cold`: Level of frequent cold (1-2 scale)
- `Dry Cough`: Level of dry cough (1-7 scale)
- `Snoring`: Level of snoring (1-4 scale)
- `Level`: Level of cancer risk (Low, Medium, or High)

## Exploratory Data Analysis
The initial exploratory data analysis involves:

1. Inspecting the data types and checking for missing values
2. Visualizing the data to identify any patterns or relationships between the features and the target variable (cancer risk level)

## Model Development
The project will involve the following steps for model development:

1. Preprocessing the data (handling missing values, encoding categorical variables, etc.)
2. Splitting the data into training and test sets
3. Training various machine learning models (e.g., logistic regression, decision trees, random forests)
4. Evaluating the models' performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score)
5. Selecting the best-performing model and fine-tuning its hyperparameters

## Results
The final results of the project, including the performance of the selected model and any key insights gained from the analysis, 
will be reported in this section.

## Usage
To run the code, you will need to have the following Python libraries installed:

- pandas
- seaborn
- matplotlib

You can then run the Jupyter Notebook file `cancer.ipynb` to reproduce the analysis.

## Conclusion
This project demonstrates the use of machine learning techniques to predict the level of cancer risk based on patient characteristics and medical history. 
The insights gained from this analysis can be useful for healthcare professionals in identifying high-risk individuals and implementing appropriate preventive measures.
