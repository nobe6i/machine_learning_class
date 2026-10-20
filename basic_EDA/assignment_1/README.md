# BAN 674: Machine Learning Data Preparation & Exploratory Analysis (Assignment 1)

This repository contains the Python notebook and detailed report documenting data preparation, exploratory data analysis (EDA), and machine learning preprocessing pipelines. The analysis evaluates two primary datasets: the **Toyota Corolla Vehicle Dataset** (for regression tasks) and the **Market Street Wine Dataset** (for classification tasks).

## Table of Contents
- [Project Overview](#project-overview)
- [Key Insights & Statistical Findings](#key-insights--statistical-findings)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Jupyter Notebook Workflow](#jupyter-notebook-code-architecture)
- [Visualizations Exported](#visualizations)
- [Dependencies](#dependencies)

---

## Project Overview

### 1. Toyota Corolla Dataset
* **Objective:** Clean, analyze, and partition vehicle metadata to prepare for structural price-prediction modeling.
* **Shape:** 1,436 vehicle observations across 39 attribute variables.
* **Attributes Evaluated:** Price, manufacturing age, mileage, horsepower, mechanical specifications (cylinders, gears, engine displacement), and factory add-ons (airbags, ABS, climate control).

### 2. Market Street Wine Dataset
* **Objective:** Perform initial data profiling, feature standardization, and baseline target distribution tracking for future multi-class classification.
* **Shape:** 178 structural samples across 14 specialized chemical parameters.
* **Attributes Evaluated:** Wine type classification (A, B, or C) mapped alongside 13 unique continuous chemical metrics including Alcohol content, Malic Acid, Ash, Magnesium, Phenols, Flavanoids, Color Intensity, Hue, and Proline.

---

## Key Insights & Statistical Findings

### Toyota Corolla Analysis
* **Price Architecture:** The average vehicle price hovers around **10,730.82** (with a median of 9,900), spanning a tight boundary from 4,350 to 32,500. Distribution visualizations reveal a significantly right-skewed layout concentrated heavily within the 7,000 to 13,000 index window.
* **Age & Manufacturing Density:** The sample space averages an active road age of **56 months**. Bulk vehicle manufacturing is constrained to the 1998–2001 timeframe, dropping severely through 2003 and 2004.
* **Structural Anomalies Detected:** Horsepower safely stabilizes near the 90–110 range. Boxplot inspections uncovered critical data anomalies, notably a vehicle recorded with a highly anomalous **16,000 cc engine displacement**, flagging potential data entry error.
* **Mechanical Dominance:** 100% of vehicles run on 4-cylinder engines. 5-speed transmissions dominate 96.8% of the dataset (1,390 units). Fuel type is highly concentrated in Petrol (88%), followed by Diesel (10.8%) and CNG (1.2%). Grey, Blue, and Red stand out as the top choices for exterior body color.

### Market Street Wine Analysis
* **Target Imbalance:** Product samples are split across three continuous classes: **Type B (71 samples)**, **Type A (59 samples)**, and **Type C (48 samples)**.
* **Chemical Benchmarks:** The collection demonstrates an average alcohol volume concentration of **13.00%**, an average magnesium metric of 99.74, and an average proline level of 746.89.
* **Outlier Variance:** Boxplots isolated severe outlier fluctuations across continuous variables, specifically within Malic Acid, Ash Alkalinity, Proanthocyanins, Color Intensity, and Proline fields.

---

## Data Preprocessing Pipeline

### 1. Feature Cleansing & Optimization
* **Variable Normalization:** Column headers were dynamically standardized by stripping white spaces and mapping structural underscores (`_`). Both datasets were verified to contain **0 missing values** across all entries.

### 2. Feature Engineering & Scaling
* **Z-Score Normalization (Toyota):** Kilometers (`KM`) was standardized using its mean and standard deviation to remove unit bias for downstream distance estimators.
  \[\text{Normalized KM} = \frac{\text{KM} - \text{mean}(\text{KM})}{\text{std}(\text{KM})}\]
* **Categorical Dummy Encoding:** The `Fuel_Type` string attribute was expanded into 3 distinct, regression-ready boolean dimensions: `Fuel_CNG`, `Fuel_Diesel`, and `Fuel_Petrol`.
* **Continuous Feature Scaling (Wine):** To counter variable scale dependencies, the high-variance `Magnesium` and `Proline` vectors were scaled to zero-centered normal profiles (`normalized_magnesium`, `normalized_proline`).

### 3. Data Partitioning Strategy
To prevent data leakage during model training and validation, the processed Toyota dataset was divided using a fixed random state (`random_state=1`):
* **Training Set:** 718 entries (50%) — *For model fitting.*
* **Validation Set:** 431 entries (30%) — *For parameter tuning.*
* **Test Set:** 287 entries (20%) — *For final model evaluation.*

---

## Jupyter Notebook Code Architecture

The underlying analysis follows a structured Python execution strategy using standard science and visualization libraries:

1. **Environment Setup:** Imports data manipulation modules (`pandas`, `numpy`), system configurations (`os`, `pathlib`), plotting engines (`matplotlib`, `seaborn`), and model validation split criteria (`sklearn`).
2. **Exploratory Visualizations:** Code cells segment continuous numerical profiles from categorical factors to build clear target distributions and highlight data anomalies.
3. **Data Transformations:** Executes inline mathematical scaling arrays alongside fast dummy matrix concatenations (`pd.get_dummies`).

---

## Visualizations

The following core exploratory plots are generated within the notebook run:
* `toyota_price_histogram.png` - Outlines the right-skewed car pricing spread.
* `toyota_year_histogram.png` - Visualizes the car manufacturing volume dips over time.
* `toyota_boxplots.png` - Tracks feature variations and highlights extreme configuration outliers.
* `wine_type_dist.png` - Illustrates the class count distribution among Wine Types A, B, and C.
* `wine_boxplots.png` - Displays chemical distributions before and after standardization.

---

## Dependencies

To run the notebook successfully, verify that your virtual python or conda environment contains the following package versions:

```bash
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
```
