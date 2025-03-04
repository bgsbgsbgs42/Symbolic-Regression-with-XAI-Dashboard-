# Symbolic Regression with XAI Dashboard

This project implements a comprehensive symbolic regression tool with an interactive web interface and explainable AI (XAI) dashboard to help users understand the discovered mathematical expressions.

## Features

### Core Symbolic Regression

- **Multi-Objective Optimization**: Finds expressions that balance accuracy and simplicity
- **Advanced Expression Discovery**: Flexible primitive sets and genetic operations
- **Incremental Learning**: Updates models when new data becomes available
- **Progressive Complexity**: Gradually increases model complexity during training

### Export Capabilities

- **Python Code**: Export expressions as Python functions for immediate use
- **C++**: Generate optimized C++ implementations for high-performance applications
- **MATLAB**: Export code compatible with MATLAB/Octave workflows
- **R**: Create functions for statistical analysis in R
- **LaTeX**: Generate publication-quality mathematical notation

### XAI Dashboard

- **Expression Analysis**: Visual breakdown of discovered expressions
- **Feature Importance**: Understand which variables matter most
- **Step-by-Step Explanations**: See how predictions are calculated
- **Interactive Predictions**: Test expressions with your own inputs
- **Reliability Metrics**: Confidence scores and statistical guarantees

### Web Interface

- **Data Upload**: CSV and Excel file support
- **Algorithm Configuration**: Fine-tune the regression process
- **Real-Time Visualization**: Watch the optimization progress
- **Cloud Deployment**: Ready for deployment on cloud platforms

## Installation

### Requirements

- Python 3.7+
- Flask
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SymPy
- DEAP (Distributed Evolutionary Algorithms in Python)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/symbolic-regression-xai.git
   cd symbolic-regression-xai
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```
   python webapp_launcher.py
   ```

The web interface will automatically open in your default browser at http://localhost:5000.

## Deployment Options

### Local Development

Run the app in development mode:
```
python webapp_launcher.py
```

### Production Mode

Run the app using a production-ready WSGI server:
```
python webapp_launcher.py --production
```

### Cloud Deployment

Generate deployment files for various cloud platforms:
```
python webapp_launcher.py --prepare-deploy
```

This creates:
- Procfile for Heroku
- app.yaml for Google App Engine
- Dockerfile for containerized environments
- requirements.txt for dependencies

## Usage Guide

### 1. Upload Data

- Click the "Upload" button and select a CSV or Excel file
- Your data should have numeric columns with headers
- The first few rows will be displayed for verification

### 2. Configure Model

- Select the target variable to predict
- Check the features you want to include in the model
- Choose an algorithm:
  - **Standard**: Basic symbolic regression
  - **Progressive Complexity**: Gradually increases expression complexity
  - **Incremental Learning**: Updates model with new data

### 3. Train Model

- Click "Start Training" to begin
- The progress bar shows optimization status
- Current metrics are updated in real-time

### 4. Explore Results

Once training is complete, you can:
- View the discovered expression
- See prediction performance metrics
- Click "Open XAI Dashboard" for in-depth analysis

### 5. XAI Dashboard

The dashboard provides multiple views:
- **Expression Structure**: Tree visualization and annotated formula
- **Feature Importance**: Bar chart of variable importance
- **Model Reliability**: Theoretical guarantees and confidence metrics
- **Interactive Prediction**: Enter values and see predictions explained
- **Feature Effects**: How each variable influences the output

### 6. Export Your Model

After discovering an expression, you can export it in multiple formats:
- **Python**: Ready-to-use Python functions for your own applications
- **C++**: High-performance implementation for production systems
- **MATLAB**: Functions compatible with MATLAB/Octave
- **R**: Statistical analysis in R
- **LaTeX**: Publication-quality mathematical notation

## Advanced Features

### Incremental Learning

The incremental learning option allows you to:
- Update models when new data becomes available
- Preserve knowledge from previous training
- Detect concept drift in evolving data
- Balance memory of old data vs. adaptation to new patterns

To use this feature:
1. Select "Incremental Learning" algorithm
2. Configure memory size and forgetting factor
3. Train on initial data
4. Add new data batches as they become available

### Progressive Complexity

This feature implements curriculum learning for symbolic regression:
1. Start with simple expressions
2. Gradually increase complexity
3. Find the optimal complexity level for your data

Configure with:
- Initial complexity: Starting complexity level
- Final complexity: Maximum complexity to explore
- Complexity steps: Number of stages in the progression

### Expression Simplification and Guarantees

After discovering an expression, the system automatically:
- Simplifies the formula for better interpretability
- Calculates theoretical guarantees for reliability
- Provides confidence metrics based on statistical tests

## API Reference

If you want to integrate the symbolic regression engine into your own application, you can use the following main classes:

- `MultiObjectiveSymbolicRegression`: Core symbolic regression with Pareto optimization
- `IncrementalSymbolicRegression`: Enables updating models with new data
- `ProgressiveComplexityRegression`: Implements curriculum learning approach
- `AdvancedSymbolicSimplification`: Simplifies expressions to more elegant forms
- `TheoreticalGuarantees`: Provides reliability metrics and statistical guarantees

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DEAP library for evolutionary computation
- SymPy for symbolic mathematics
- Flask for the web framework
- Bootstrap for the UI components
