# Gold Forex Trading Signal Prediction Using Machine Learning

## Overview

This project aims to develop a machine learning model to predict **buy** and **sell** signals in the **gold forex market**. By utilizing historical price data and technical indicators, the project explores various machine learning models and applies advanced feature engineering techniques, including **polynomial features** and **Principal Component Analysis (PCA)**, to improve the model's predictive performance.

## Project Phases

### Phase 1: Data Preprocessing and Feature Engineering
- **Data Source**: Historical gold price data retrieved every 30 minutes from Yahoo Finance.
- **Technical Indicators**: Various indicators such as RSI, MACD, Bollinger Bands, and Moving Averages were calculated to capture market conditions.
- **Target Variable (Action)**: The price change is categorized into a range of -6 to 6, representing the magnitude of price movements.
- **Feature Shifting**: Technical indicators were shifted to align with the next period's open, close, high, and low prices.

### Phase 2: Model Selection and Hyperparameter Tuning
- **Polynomial Features**: A polynomial transformation (degree 3) was applied to capture complex feature interactions, resulting in 26,000 features.
- **PCA**: PCA was used to reduce the feature set to 79 components, retaining 99% of the variance in the data.
- **Model Selection**: The following models were tuned using GridSearchCV:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
  - SVM
  - Multi-Layer Perceptron (MLP)

### Phase 3: Evaluation and Results
- **ROC Curve**: ROC curves and AUC were calculated for all models.
- **Performance Metrics**: Accuracy, F1-Score, Precision, Recall, and Confusion Matrices were analyzed for each model.
- **Final Ranking**: Models were ranked based on their suitability for the trading context, with a focus on accuracy and the ability to identify the minority class (buy/sell signals).

## Repository Structure

```bash
├── data                                        # Data files used in the project
│   ├── [RAW-DATA.csv]                          # Prepared raw data file
├── scripts                                     # Python scripts for data processing and model training
│   ├── [Multiple Models with Grid Search.ipynb]# Script to retrieve and preprocess data
│   ├── [Different_models.ipynb]                # Script for model training and evaluation
├── notebooks                                   # Jupyter notebooks for experimentation
│   ├── [FX_gold_trade.ipynb]
├── [README.md]                                 # Project overview and instructions
└── [LICENSE](LICENSE)                          # License for the project
```
[RAW-DATA.csv](RAW-DATA.csv)
[Multiple Models with Grid Search.ipynb](Multiple%20Models%20with%20Grid%20Search.ipynb)
[Different_models.ipynb](scripts/Different_models.ipynb)
[FX_gold_trade.ipynb](notebooks/FX_gold_trade.ipynb)
[README.md](README.md)
[LICENSE](LICENSE)



## Key Features

- **Feature Engineering**: Includes the use of advanced technical indicators and polynomial feature transformations.
- **Dimensionality Reduction**: PCA was applied to reduce the computational complexity.
- **Model Comparison**: Multiple machine learning models were compared and ranked for predictive performance.
- **Imbalanced Data Handling**: Techniques such as SMOTE and class weighting were used to handle imbalanced data.

## Results

- **Top Model**: Random Forest achieved the best balance between accuracy and the identification of minority class signals.
- **ROC Curves**: ROC curves and AUC scores provide insights into each model's ability to separate buy/sell signals.

## Future Work

- **Hyperparameter Tuning**: Further fine-tuning of hyperparameters could improve model performance.
- **Deep Learning Models**: Exploring more complex architectures such as LSTMs or CNNs for time-series data.
- **Real-Time Trading**: Integration with live data feeds for real-time trading signal generation.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Acknowledgements

Special thanks to the contributors who helped with this project, and to Yahoo Finance for providing the historical gold price data.
