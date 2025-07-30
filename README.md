# Plan Success Prediction & Recommendation System (MATLAB)

## Overview

This MATLAB project predicts the success probability of personal plans and offers improvement suggestions based on historical data. It combines machine learning (Decision Tree) and deep learning (LSTM) models in a hybrid approach to forecast success rates and generate actionable feedback.

## Key Features

- Train models from historical plan data (`plan_*.csv`)
- Predict success rates for new plans (`new_plan.csv`)
- Provide improvement suggestions using averages and similar case patterns
- Display prediction results with formatted text and a donut chart
- Report model performance (accuracy and MSE)

## Project Structure

project_root/
├── plan_*.csv             # Training plan data (multiple files allowed)
├── new_plan.csv           # New plan data for prediction
├── planPredictor.m        # Main executable MATLAB script
└── README.md              # This documentation file

## Input Data Format

### Training Data (`plan_*.csv`)

| Category | SubCategory | Priority | TargetTime | ActualTime | TaskSuccess | PlanSuccess |
|----------|-------------|----------|------------|------------|-------------|-------------|
| string   | string      | double   | double     | double     | 0 or 1      | 0 or 1      |

### New Plan Data (`new_plan.csv`)

| Category | SubCategory | Priority | TargetTime |
|----------|-------------|----------|------------|
| string   | string      | double   | double     |

## Model Description

- **Decision Tree**: Handles categorical and numerical features to predict binary plan success.
- **LSTM (Long Short-Term Memory)**: Uses sequential inputs for regression-based success probability.
- **Hybrid Ensemble**: Combines both models using the formula:

  Final Score = 0.7 × DecisionTree + 0.3 × LSTM


## How to Run

1. Place training CSV files as `plan_*.csv` in the `trainFolder`.
2. Prepare a `new_plan.csv` file with new plan data.
3. Run the MATLAB script:

   matlab run("planPredictor.m");


4. View output in the command window and generated visualizations.

## Output Example
### Console Output
Item 1: Success Probability = 72.4%
Suggestion: Reduce the target time.

### Graph Output

- Text summaries per item
- Donut chart representing overall success probability

## Model Evaluation
- Binary Classification Accuracy
- LSTM Regression MSE
- Threshold: Success if probability > 0.5

## Future Improvements
- Add more features (e.g., day of week, time slot)
- Weighted scoring by plan category
- Interactive user feedback integration (GUI)

## Developer Notes
This system was designed to enhance personal productivity through predictive modeling and tailored feedback. The hybrid model balances structured logic and sequential pattern learning for improved decision support.
