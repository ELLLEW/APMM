# Plan Success Prediction & Recommendation System (MATLAB)

## Overview

This MATLAB project predicts the success probability of personal plans and offers improvement suggestions based on historical data. It combines machine learning (Decision Tree) and deep learning (LSTM) models in a hybrid approach to forecast success rates and generate actionable feedback.

## Key Features

- Train models from historical plan data (`plan.zip`, containing multiple `plan_*.csv` files)
- Predict success rates for new plans (`new_plan.csv`)
- Provide improvement suggestions using averages and similar case patterns
- Display prediction results with formatted text and a donut chart
- Report model performance (accuracy and MSE)

## Project Structure

project_root
- APMM.m             # Main executable MATLAB script
- README.md          # This documentation file
- new_plan.csv       # New plan data for prediction
- plan.zip           # Training plan data (contains plan_1.csv, plan_2.csv, etc.)


## Input Data Format

### Training Data (`plan_*.csv` inside plan.zip)

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

1. Extract `plan.zip` into a folder (e.g., `trainFolder`) so that you have multiple `plan_*.csv` files.
2. Place `new_plan.csv` (or a csv file what you want to test) in the same or a known location.
3. Open MATLAB, navigate to the project folder, and run:

   matlab
   run('APMM.m')


4. View results in the Command Window and graphical output window.

## Output Example

### Console Output

Item 1: Success Probability = 72.4%
<br>Suggestion: Reduce the target time.

### Graph Output

- Text summaries per item
- Donut chart representing overall success probability

## Model Evaluation

- Binary Classification Accuracy (for Decision Tree)
- Regression Mean Squared Error (for LSTM)
- Success threshold: probability > 0.5

## Future Improvements

- Add more contextual features (e.g., day of week, time slot)
- Weighted scoring based on plan category
- Interactive user feedback integration (GUI)

## Developer Notes

This system was designed to enhance personal productivity through predictive modeling and tailored feedback. The hybrid model balances structured logic and sequential pattern learning for improved decision support.
