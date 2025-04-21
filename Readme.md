# SVM Parameter Optimization for Letter Recognition

This project performs Support Vector Machine (SVM) parameter optimization using `GridSearchCV` on the [Letter Recognition dataset]. It evaluates different SVM configurations to classify capital letters based on statistical image features.

---

## Dataset

**File**: `letter-recognition.csv`  
**Source**: UCI Machine Learning Repository  
**Columns**:
- `letter`: The target class (A-Z)
- 16 numerical features (e.g., `x-box`, `y-box`, `width`, `high`, etc.)

---

## Approach

1. **Data Preprocessing**:
   - The target labels (letters) are encoded to integers using `LabelEncoder`.
   - Features are standardized using `StandardScaler`.

2. **Model**: `SVR` (Support Vector Regression) is used for multi-class classification by rounding and clipping predicted outputs.

3. **Hyperparameter Tuning**:
   - A pipeline is created with a scaler and SVR model.
   - A grid search is performed with 3-fold cross-validation over the following parameters:
     - `kernel`: `linear`, `rbf`, `poly`
     - `C`: `0.1`, `1`, `10`
     - `epsilon`: `0.01`, `0.1`, `0.5`

4. **Experimentation**:
   - The experiment is repeated 10 times with different random splits (`random_state = 0 to 9`).
   - For each trial, the best model is selected based on cross-validation.
   - Accuracy is computed on the test set by rounding predictions to the nearest class index.

5. **Convergence Analysis**:
   - For each best model, incremental training is done with increasing portions of the training data (1% to 100%) to analyze convergence behavior.

6. **Results**:
   - Best hyperparameters and accuracy for each trial are saved to `svm_results.csv`.
   - Accuracy convergence curves are stored in memory (`all_convergences`).

---

## Output

- **CSV File**: `svm_results.csv` containing:
  - `Sample`: Trial ID
  - `Accuracy`: Test accuracy (rounded predictions)
  - `Kernel`: Best SVM kernel
  - `C`: Best regularization parameter
  - `Epsilon`: Best epsilon-insensitive loss threshold

---

Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib
