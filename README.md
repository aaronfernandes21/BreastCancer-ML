# Breast Tumor Identifier

## Description

This project uses machine learning classifiers to predict whether a tumor is benign or malignant based on various features such as cell size, shape, and texture. The goal is to implement and evaluate three classification models:
- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

The models are trained and tested on the Breast Cancer Wisconsin dataset. Their performance is compared using accuracy, confusion matrices, and classification reports. The project also visualizes the accuracy comparison and feature importance for the Random Forest classifier.

## Dataset Implemented

The dataset used in this project is the **Breast Cancer Wisconsin dataset**. It includes the following columns:
- **diagnosis**: Target variable with values 'M' (Malignant) and 'B' (Benign).
- Various features related to tumor characteristics like `radius_mean`, `texture_mean`, `perimeter_mean`, etc.

## Requirements

Before running this project, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Project Structure

```
BreastCancerMLProject/
│
├── data/
│   └── data.csv          # Raw dataset
├── src/
│   └── main.py           # Main script for training, evaluation, and visualization
├── README.md             # Project README (this file)
└── requirements.txt      # List of required Python packages
```

## Steps

### 1. Load and Preprocess the Data

- The data is loaded from a CSV file.
- The unnecessary columns (`Unnamed: 32` and `id`) are removed.
- The target variable, `diagnosis`, is mapped to numeric values ('M' -> 1, 'B' -> 0).
- Missing values are handled, if any.

### 2. Train and Test Classifiers

Three classifiers are used:
- **Support Vector Machine (SVM)** with a linear kernel.
- **Random Forest** with optimized hyperparameters.
- **K-Nearest Neighbors (KNN)** with an optimized number of neighbors.

These classifiers are trained on the training dataset and evaluated on the testing dataset.

### 3. Model Evaluation

The following evaluation metrics are used:
- **Accuracy**: The proportion of correct predictions.
- **Confusion Matrix**: Provides insights into true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score for each class (Malignant and Benign).

### 4. Visualizations

- **Confusion Matrices**: For each model (SVM, Random Forest, and KNN).
- **Accuracy Comparison Bar Chart**: Compares the accuracy of the three models.
- **Feature Importance (Random Forest)**: Displays the most important features based on the Random Forest model.

### 5. Cross-Validation

The models are evaluated with 5-fold cross-validation to ensure the results are reliable and not overfitted to the training data.

### Example Output

- **Accuracy**:
  - SVM: 97%
  - Random Forest: 98%
  - KNN: 96%

- **Classification Report (for one model)**:
    ```
    precision    recall  f1-score   support

        0       0.99      0.99      0.99       71
        1       0.98      0.98      0.98       46

    accuracy                           0.99       117
    macro avg       0.99      0.99      0.99       117
    weighted avg    0.99      0.99      0.99       117
    ```

- **Confusion Matrix** (Example for Random Forest):
  ```
    Predicted  0  1
    Actual
    0          71  0
    1           1  45
  ```

## How to Run

1. **Download the Dataset**:
   - Place the `data.csv` file in the `data/` folder.

2. **Run the Main Script**:
   - Navigate to the `src/` folder and run the script:
     ```bash
     python main.py
     ```

   - This will train and evaluate the models, print the results, and display the confusion matrices and accuracy comparison plots.

## Notes

- The dataset is assumed to be in a CSV format and is expected to have a specific structure. Ensure that the dataset is properly formatted before running the script.
- The project may require adjustments if used with other datasets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

