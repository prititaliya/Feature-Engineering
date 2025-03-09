# Feature Engineering

This repository contains code and examples for feature engineering, focusing on encoding techniques and feature selection. The goal is to preprocess data effectively for machine learning models.

## Encoding Techniques

### 1. **Ordinal Encoding**
   - **Use Case**: Applied to ordinal categorical data where the categories have a natural order.
   - **Example**: 
     - `review`: Poor, Average, Good
     - `education`: School, UG, PG
   - **Implementation**:
     ```python
     from sklearn.preprocessing import OrdinalEncoder
     oe = OrdinalEncoder(categories=[["Poor","Average","Good"], ['School','UG','PG']])
     x_scaled = oe.fit_transform(x)
     ```

### 2. **Label Encoding**
   - **Use Case**: Applied to the target variable `y` when it is categorical.
   - **Implementation**:
     ```python
     from sklearn.preprocessing import LabelEncoder
     le = LabelEncoder()
     y_scaled = le.fit_transform(y)
     ```

### 3. **One-Hot Encoding**
   - **Use Case**: Applied to nominal categorical data where no natural order exists.
   - **Example**: 
     - `brand`: Maruti, Skoda, Honda, etc.
     - `fuel`: Diesel, Petrol
     - `owner`: First Owner, Second Owner, etc.
   - **Implementation**:
     ```python
     from sklearn.preprocessing import OneHotEncoder
     ohe = OneHotEncoder(drop='first')
     df_to_encode = ohe.fit_transform(df_to_encode).toarray()
     ```

## Feature Selection

### Removing Low-Threshold Brands
   - **Rationale**: Some brands may have very few samples in the dataset, which can lead to overfitting or poor generalization. It's often beneficial to remove brands with a low count (e.g., less than a certain threshold).
   - **Implementation**:
     ```python
     brand_counts = cars['brand'].value_counts()
     threshold = 100  # Example threshold
     low_threshold_brands = brand_counts[brand_counts < threshold].index
     cars_filtered = cars[~cars['brand'].isin(low_threshold_brands)]
     ```

## Usage

1. **Ordinal Encoding**: Use for ordinal categorical features in `X`.
2. **Label Encoding**: Use for the target variable `y`.
3. **One-Hot Encoding**: Use for nominal categorical features in `X`.
4. **Feature Selection**: Remove low-threshold brands to improve model performance.

## Future Improvements
- Experiment with different thresholds for removing low-frequency brands.
- Explore other feature selection techniques such as Recursive Feature Elimination (RFE) or feature importance from tree-based models.

## Author
- **Prit Italiya**
