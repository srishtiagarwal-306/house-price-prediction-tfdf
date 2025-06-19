# 🏡 House Price Prediction using TensorFlow Decision Forests

This project focuses on predicting house prices using **TensorFlow Decision Forests (TFDF)**. We apply a Random Forest regressor on structured data from the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

---

## 🔍 Project Overview

📈 Real estate prices are influenced by multiple factors like location, house size, quality, year built, etc. This notebook leverages tree-based models to automatically capture such patterns in the data without extensive feature engineering.

💡 **Key Idea:** Use **TFDF** to bypass preprocessing like one-hot encoding or scaling and still get excellent results with minimal code.

---

## 📁 Project Structure

```bash
house-price-prediction-tfdf/
├── housepp.ipynb               # Main notebook with model training and prediction
├── requirements.txt            # List of Python dependencies
├── README.md                   # This file
├── dataset/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset (no SalePrice)
│   └── sample_submission.csv   # Submission format from Kaggle
└── submission.csv              # Output file for Kaggle submission
```

---

## 📊 Dataset Description

* **Source:** Kaggle - [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* **Target:** `SalePrice` (Continuous regression variable)
* **Training records:** 1460
* **Test records:** 1459
* **Key Features:**

  * `OverallQual`, `GrLivArea`, `TotalBsmtSF`
  * `YearBuilt`, `GarageArea`, `Neighborhood`
  * `LotArea`, `HouseStyle`, `ExterQual`

---

## 🎯 Goals

* Load and process the training and test datasets
* Train a robust **Random Forest model** using TFDF
* Evaluate feature importance
* Predict house prices on the test set
* Submit results in proper CSV format

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction-tfdf.git
cd house-price-prediction-tfdf
```

### 2. Create a virtual environment

```bash
python3 -m venv tfenv
source tfenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset files

Place the following Kaggle-provided files inside the `dataset/` folder:

* `train.csv`
* `test.csv`
* `sample_submission.csv`

### 5. Launch the notebook

```bash
jupyter notebook housepp.ipynb
```

---

## 🧠 Model Used

### ✅ TensorFlow Decision Forests

We used:

```python
tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
```

Advantages:

* Handles **categorical and numerical features** natively
* No need for extensive preprocessing
* High accuracy with minimal tuning

---

## 🔍 Feature Importance (Sample)

The model helps us identify which features most impact price predictions:

| Rank | Feature       | Importance |
| ---- | ------------- | ---------- |
| 1    | `OverallQual` | ⭐️⭐️⭐️⭐️⭐️ |
| 2    | `GrLivArea`   | ⭐️⭐️⭐️⭐️   |
| 3    | `GarageCars`  | ⭐️⭐️⭐️     |
| 4    | `TotalBsmtSF` | ⭐️⭐️⭐️     |
| 5    | `YearBuilt`   | ⭐️⭐️⭐️     |

📊 These insights are printed in the notebook using:

```python
inspector = rf.make_inspector()
inspector.features()
```

---

## 📷 Visuals & Output Samples

* 📈 Prediction snapshot:

```python
sample_submission_df['SalePrice'].head()
```

* 📝 Final submission:

```python
sample_submission_df.to_csv("submission.csv", index=False)
```

* ✅ Output example:

| Id   | SalePrice |
| ---- | --------- |
| 1461 | 208000    |
| 1462 | 181000    |
| 1463 | 223000    |

---

## 🚀 Future Enhancements

* Add validation set and RMSLE scoring
* Train other models (XGBoost, LightGBM, CatBoost)
* Hyperparameter tuning with Optuna or KerasTuner
* Streamlit dashboard for live predictions
* Feature engineering: total area, house age, condition scores

---

## 🙌 Acknowledgments

* [Gusthema’s TFDF Notebook](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf)
* TensorFlow Decision Forests by Google
* Kaggle House Prices Dataset

---

## 📝 License

This project is for educational purposes. If you reuse any part, please provide attribution.

---
