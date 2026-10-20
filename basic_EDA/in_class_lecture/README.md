# West Roxbury Housing Price Prediction & Data Pipeline

This repository contains a structured data preprocessing and machine learning workflow executed within a Jupyter Notebook to explore, clean, and model housing values in the West Roxbury neighborhood using linear regression.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Specifications](#dataset-architecture)
- [Data Cleaning & Engineering Pipeline](#data-cleaning--engineering-pipeline)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Installation & Getting Started](#installation--getting-started)
- [Notebook Structure](#notebook-structure)

---

## Project Overview
The primary objective of this project is to build a predictive model for property values (`TOTAL_VALUE`) based on localized structural attributes, real estate metrics, and remodeling histories. 

---

## Dataset Architecture
The core dataset utilized is `WestRoxbury.csv`, which consists of **5,802 property records** across **14 evaluation columns**:

*   **TOTAL_VALUE:** Total assessed value of the property (Target Variable)
*   **TAX:** Tax amount assessed on the property
*   **LOT_SQFT:** Total square footage of the lot
*   **YR_BUILT:** Year the structure was built
*   **GROSS_AREA / LIVING_AREA:** Gross versus net functional living spaces
*   **FLOORS / ROOMS / BEDROOMS:** Physical dimensions and layout counts
*   **FULL_BATH / HALF_BATH / KITCHEN / FIREPLACE:** Amenity and utility tracking metrics
*   **REMODEL:** Categorical indicator representing the remodel history status (`None`, `Old`, `Recent`)

---

## Data Cleaning & Engineering Pipeline

### 1. Header & String Optimization
* Strips trailing whitespaces from the column labels.
* Replaces internal blank spaces with unified pythonic underscores (`_`) to optimize dataframe call syntaxes (e.g., transforming `TOTAL VALUE ` into `TOTAL_VALUE`).

### 2. Missing Value Imputation
* Detects rows containing null markers (`NaN`).
* Employs an imputation strategy that replaces missing value parameters within structural metrics (e.g., `BEDROOMS`) with the computed **median value** of the valid records, preserving data volume without list-wise deletion biases.

### 3. Categorical Dummy Encoding
* Converts the textual `REMODEL` object feature into a clean mathematical matrix using **One-Hot Encoding** (`pd.get_dummies`).
* Drops the first category level to avoid multicollinearity (the "dummy variable trap"), generating binary flags: `REMODEL_Old` and `REMODEL_Recent`.

### 4. Data Partitioning & Feature Validation
* Isolates `TOTAL_VALUE` as the dependent vector (y) and strips `TAX` along with the target from the predictor matrix (X) to avoid data leakages.
* Segmented using a randomized 60/40 validation split (`random_state=1`):
  * **Training Set:** 3,481 records (60% of the sample space)
  * **Validation Set:** 2,321 records (40% of the sample space)

---

## Machine Learning Workflow

The notebook trains a **scikit-learn Linear Regression** model on the partitioned training data to predict real estate valuations.

### Performance Tracking & Diagnostics
The script tracks regression errors by evaluating model behaviors against real validation data using standard diagnostic frameworks:
* **Mean Error (ME)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **Mean Percentage Error (MPE)**
* **Mean Absolute Percentage Error (MAPE)**

### Out-of-Sample Predictions
An integrated validation array is provided at the bottom of the script to pass entirely new, unseen property data frames directly to the trained estimator, outputting immediate valuation projections:
```python
# Pass new structural metadata directly to the trained regression vector
predictions = model.predict(new_data)
```

---

## Installation & Getting Started

### Prerequisites
Ensure your local Python kernel or environment has the core data science stack installed:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Running the Notebook
1. Clone this repository to your local directory.
2. Ensure `WestRoxbury.csv` is placed in the root folder alongside the notebook.
3. Open a terminal and launch the interface:
   ```bash
   jupyter notebook
   ```
4. Run all cells sequentially to witness the full preprocessing and modeling execution.

---

## Notebook Structure
* **Section 1:** Library imports (`pandas`, `numpy`, `sklearn`, `matplotlib`) and package installs.
* **Section 2:** Data ingestion and initial shape checking (`.shape`, `.head()`).
* **Section 3:** Header parsing and trailing whitespace removal.
* **Section 4:** Null vector extraction, subset filtering, and median value imputation.
* **Section 5:** Continuous feature scaling templates (Z-score standardization and MinMax mapping architectures).
* **Section 6:** Dummy array conversion and train/test splits.
* **Section 7:** Model initialization, regression line fitting, residual error graphing, and batch data prediction.
