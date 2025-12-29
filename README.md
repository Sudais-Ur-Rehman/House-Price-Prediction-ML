# 🏡 House Price Prediction ML Project

Machine Learning project for **House Price Prediction using Python, scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn**.  
Includes **data preprocessing, feature engineering, exploratory data analysis (EDA), model training, hyperparameter tuning, and regression evaluation metrics**.  
This project demonstrates end-to-end **machine learning for real estate price prediction** using the California Housing dataset.

## 📂 Files Included
- `Housing_Price_Prediction.ipynb` → Complete notebook with EDA, modeling, and evaluation
- `housing.csv` → Dataset used for training and testing
- `Data Correlation.png` → Heatmap of feature correlations
- `Histplot of Data.png` → Distribution of numerical features

## 📊 Dataset
- Source: [California Housing Prices – Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- Features:
  - `longitude`, `latitude` → Location coordinates
  - `housingMedianAge` → Age of houses
  - `totalRooms`, `totalBedrooms`, `population`, `households` → Block-level housing stats
  - `medianIncome` → Median income per block
  - `medianHouseValue` → Target variable (house price)
  - `oceanProximity` → Categorical location feature

## 📈 Exploratory Data Analysis (EDA)
- Checked for missing values, duplicates, and feature distributions
- Visualized histograms, boxplots, scatter plots, and heatmaps
- Key insights:
  - **Median Income** is the strongest predictor of house value
  - **Ocean Proximity** and **location coordinates** affect pricing
  - **Raw totals** (rooms, bedrooms, population) are weak predictors
  - **Engineered ratios** (rooms/household, bedrooms/room, population/household) improve model performance
  - **Target variable** is capped at $500k — log transformation applied

## 🛠️ Feature Engineering
- Imputed missing values using median strategy
- Created new features:
  - `rooms_per_household`
  - `bedrooms_per_rooms`
  - `population_per_household`
- Applied `np.log1p()` to target variable for better regression fit

## 🔄 Data Preprocessing
- Train-test split (80/20)
- Pipelines for:
  - Categorical encoding (`OneHotEncoder`)
  - Scaling (`StandardScaler`)
  - Power transformation for skewed features
- Combined using `ColumnTransformer`

## 🤖 Model Training
- **Linear Regression** pipeline trained and evaluated
- Metrics: R² Score, MAE, RMSE
- Cross-validation with `cross_validate`

## 🌲 Random Forest with Hyperparameter Tuning
- Tuned parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- Best model selected via `RandomizedSearchCV`
- Predictions inverse-transformed to dollar scale

## 📊 Final Results

| Metric              | Linear Regression | Random Forest (Best Model) |
|---------------------|-------------------|----------------------------|
| R² Score (log)      | 0.76              | **0.831**                  |
| MAE ($)             | ~$45,000          | **$32,588**                |
| RMSE ($)            | ~$62,000          | **$51,216**                |

### ✅ Interpretation
- Random Forest explains **83% of variance** in house prices (log scale).  
- Average prediction error is **$32k**, showing strong accuracy for real estate valuation.  
- RMSE of **$51k** indicates predictions are close to actual values.  
- Random Forest clearly outperforms Linear Regression, proving the importance of non-linear models in **machine learning for housing price prediction**.

## 🚀 Future Work
- Deploy model using **FastAPI** for real-time predictions
- Build interactive frontend with **Streamlit** or **Gradio**
- Explore advanced models (**XGBoost, Gradient Boosting, LightGBM**)
- Add **SHAP values** or permutation-based feature importance for interpretability

## 🙌 Author
**Sudais Ur Rehman**  
Computer Science student exploring **Data Science, Machine Learning, and AI deployment**  
Connect on [LinkedIn](https://www.linkedin.com/in/sudais-ur-rehman/)
