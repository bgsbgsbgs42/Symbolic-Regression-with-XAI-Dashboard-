"""
Web Application for Symbolic Regression with XAI Dashboard and Enhanced Export Capabilities

This module implements:
1. A Flask web application for symbolic regression
2. An interactive XAI dashboard to explain discovered expressions
3. Real-time visualization of the regression process
4. Data upload and parameter configuration
5. Enhanced export options for discovered expressions (Python, C++, LaTeX, MATLAB)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json
import uuid
import time
from datetime import datetime
from threading import Thread
from queue import Queue
import sympy as sp
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template, request, jsonify, session, send_file, Response

# Import symbolic regression modules
# Note: These modules should be available in the same directory
from multi_objective_symbolic_regression import MultiObjectiveSymbolicRegression
from incremental_progressive import IncrementalSymbolicRegression, ProgressiveComplexityRegression
from advanced_simplification_guarantees import AdvancedSymbolicSimplification, TheoreticalGuarantees


# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to store background tasks and results
task_queue = Queue()
task_results = {}
active_tasks = {}


# Background task handler
def background_worker():
    """Background worker to process tasks from the queue"""
    while True:
        task_id, function, args, kwargs = task_queue.get()
        
        try:
            # Update task status
            active_tasks[task_id]['status'] = 'running'
            
            # Execute the function
            result = function(*args, **kwargs)
            
            # Store the result
            task_results[task_id] = result
            active_tasks[task_id]['status'] = 'completed'
            active_tasks[task_id]['completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            # Handle errors
            active_tasks[task_id]['status'] = 'failed'
            active_tasks[task_id]['error'] = str(e)
        
        # Mark task as done
        task_queue.task_done()


# Start background worker
worker_thread = Thread(target=background_worker, daemon=True)
worker_thread.start()


def parse_uploaded_data(file):
    """
    Parse uploaded CSV or Excel file into a pandas DataFrame
    
    Args:
        file: Uploaded file object
        
    Returns:
        data: Pandas DataFrame
        message: Status message
    """
    filename = file.filename
    
    try:
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Parse based on file extension
        if filename.endswith('.csv'):
            data = pd.read_csv(temp_path)
        elif filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(temp_path)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel files."
        
        # Check if data is empty
        if data.empty:
            return None, "The uploaded file contains no data."
        
        return data, "Data uploaded successfully."
        
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def train_symbolic_regression(data, target_column, feature_columns, params, task_id):
    """
    Train symbolic regression model with the specified parameters
    
    Args:
        data: Pandas DataFrame with the data
        target_column: Name of the target column
        feature_columns: List of feature column names
        params: Dictionary of parameters
        task_id: Unique task ID
        
    Returns:
        results: Dictionary with model results
    """
    # Extract parameters
    algorithm = params.get('algorithm', 'standard')
    test_size = float(params.get('test_size', 0.2))
    standardize = params.get('standardize', True)
    random_state = int(params.get('random_state', 42))
    population_size = int(params.get('population_size', 300))
    generations = int(params.get('generations', 30))
    
    # Prepare data
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize if requested
    if standardize:
        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_test = feature_scaler.transform(X_test)
        
        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        feature_scaler = None
        target_scaler = None
    
    # Create and fit model based on selected algorithm
    start_time = time.time()
    
    # Create the model
    if algorithm == 'standard':
        model = MultiObjectiveSymbolicRegression(
            population_size=population_size,
            generations=generations,
            feature_names=feature_columns,
            verbose=True
        )
    elif algorithm == 'incremental':
        base_model = MultiObjectiveSymbolicRegression
        model = IncrementalSymbolicRegression(
            base_regressor=base_model,
            memory_size=params.get('memory_size', 1000),
            forgetting_factor=float(params.get('forgetting_factor', 0.9)),
            verbose=True
        )
    elif algorithm == 'progressive':
        base_model = MultiObjectiveSymbolicRegression
        model = ProgressiveComplexityRegression(
            base_regressor=base_model,
            initial_complexity=int(params.get('initial_complexity', 2)),
            final_complexity=int(params.get('final_complexity', 10)),
            complexity_steps=int(params.get('complexity_steps', 5)),
            verbose=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Generate progress updates
    progress_updates = []
    
    def progress_callback(step, total_steps, metrics=None):
        """Callback function to track progress"""
        if metrics is None:
            metrics = {}
        
        progress = {
            'step': step,
            'total_steps': total_steps,
            'percentage': int(100 * step / total_steps),
            'metrics': metrics,
            'time': time.time() - start_time
        }
        progress_updates.append(progress)
        
        # Update task status
        active_tasks[task_id]['progress'] = progress['percentage']
        if 'metrics' in progress and progress['metrics']:
            active_tasks[task_id]['current_metrics'] = progress['metrics']
    
    # Add progress callback to model if supported
    if hasattr(model, 'set_callback'):
        model.set_callback(progress_callback)
    
    # Fit the model
    model.fit(X_train, y_train, feature_names=feature_columns)
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Inverse transform if standardized
    if standardize:
        y_train = target_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_pred = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
        y_test_pred = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Get the best expression
    best_expression = model.get_best_expression() if hasattr(model, 'get_best_expression') else "Unknown"
    
    # Apply advanced simplification
    simplifier = AdvancedSymbolicSimplification(
        feature_names=feature_columns,
        complexity_measure='cognitive',
        rationalize_constants=True,
        verbose=False
    )
    
    try:
        # Simplify the expression
        simplified_expr = simplifier.simplify(best_expression)
        simplified_expression = str(simplified_expr)
    except:
        simplified_expression = best_expression
    
    # Generate theoretical guarantees
    guarantor = TheoreticalGuarantees(
        feature_names=feature_columns,
        verbose=False
    )
    
    try:
        # Evaluate guarantees
        guarantee_results = guarantor.evaluate_model_guarantees([model], X_test, y_test)
        guarantee_report = guarantor.generate_guarantee_report(0)
    except:
        guarantee_report = None
    
    # Create feature importance analysis
    feature_importance = analyze_feature_importance(best_expression, feature_columns)
    
    # Generate visualizations
    visualizations = generate_visualizations(model, X, y, X_train, y_train, X_test, y_test, 
                                          y_train_pred, y_test_pred, feature_columns)
    
    # Generate export formats
    export_formats = generate_export_formats(model, simplified_expression, feature_columns)
    
    # Store results
    results = {
        'algorithm': algorithm,
        'expression': best_expression,
        'simplified_expression': simplified_expression,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'execution_time': time.time() - start_time,
        'progress_updates': progress_updates,
        'feature_importance': feature_importance,
        'visualizations': visualizations,
        'guarantee_report': guarantee_report,
        'export_formats': export_formats,
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    return results


def generate_export_formats(model, expression, feature_columns):
    """
    Generate different export formats for the discovered expression
    
    Args:
        model: Trained model
        expression: String representation of the expression
        feature_columns: List of feature column names
        
    Returns:
        export_formats: Dictionary with different export formats
    """
    export_formats = {}
    
    try:
        # Parse expression using sympy
        sympy_expr = sp.sympify(expression)
        
        # 1. LaTeX format
        export_formats['latex'] = sp.latex(sympy_expr)
        
        # 2. Python function
        python_code = f"""
import numpy as np

def predict(X):
    \"\"\"
    Make predictions using the discovered symbolic expression.
    
    Args:
        X: Input features as array-like of shape (n_samples, {len(feature_columns)})
           or a single sample of shape ({len(feature_columns)},)
           
    Returns:
        y_pred: Predicted values
    \"\"\"
    # Convert to numpy array
    X = np.array(X)
    
    # Handle both single samples and multiple samples
    if X.ndim == 1:
        # Single sample
        return predict_single(X)
    else:
        # Multiple samples
        return np.array([predict_single(x) for x in X])
    
def predict_single(x):
    \"\"\"Predict for a single sample\"\"\"
    # Assign variables
    {chr(10).join([f"{feature} = x[{i}]" for i, feature in enumerate(feature_columns)])}
    
    # Compute prediction
    return {expression}
"""
        export_formats['python'] = python_code
        
        # 3. C++ function
        # Replace Python-specific functions with C++ equivalents
        cpp_expr = expression
        cpp_expr = cpp_expr.replace('np.sin', 'sin')
        cpp_expr = cpp_expr.replace('np.cos', 'cos')
        cpp_expr = cpp_expr.replace('np.exp', 'exp')
        cpp_expr = cpp_expr.replace('np.log', 'log')
        cpp_expr = cpp_expr.replace('np.sqrt', 'sqrt')
        cpp_expr = cpp_expr.replace('np.power', 'pow')
        cpp_expr = cpp_expr.replace('np.abs', 'abs')
        
        cpp_code = f"""
#include <vector>
#include <cmath>

/**
 * Make predictions using the discovered symbolic expression.
 * 
 * @param X Input features as vector of vectors, shape (n_samples, {len(feature_columns)})
 * @return Predicted values
 */
std::vector<double> predict(const std::vector<std::vector<double>>& X) {{
    std::vector<double> predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {{
        predictions.push_back(predict_single(sample));
    }}
    
    return predictions;
}}

/**
 * Predict for a single sample
 * 
 * @param x Single sample as vector, shape ({len(feature_columns)})
 * @return Predicted value
 */
double predict_single(const std::vector<double>& x) {{
    // Assign variables
    {"".join([f"double {feature} = x[{i}];\n    " for i, feature in enumerate(feature_columns)])}
    
    // Compute prediction
    return {cpp_expr};
}}
"""
        export_formats['cpp'] = cpp_code
        
        # 4. MATLAB function
        matlab_expr = expression
        matlab_expr = matlab_expr.replace('np.sin', 'sin')
        matlab_expr = matlab_expr.replace('np.cos', 'cos')
        matlab_expr = matlab_expr.replace('np.exp', 'exp')
        matlab_expr = matlab_expr.replace('np.log', 'log')
        matlab_expr = matlab_expr.replace('np.sqrt', 'sqrt')
        matlab_expr = matlab_expr.replace('np.power', '^')
        matlab_expr = matlab_expr.replace('np.abs', 'abs')
        
        matlab_code = f"""
function y = predict(X)
% PREDICT Make predictions using the discovered symbolic expression.
%
%   Args:
%       X: Input features as matrix of shape (n_samples, {len(feature_columns)})
%          or a single sample of length {len(feature_columns)}
%   
%   Returns:
%       y: Predicted values

    % Handle both single samples and multiple samples
    if isvector(X) && length(X) == {len(feature_columns)}
        % Single sample
        y = predict_single(X);
    else
        % Multiple samples
        [n_samples, ~] = size(X);
        y = zeros(n_samples, 1);
        for i = 1:n_samples
            y(i) = predict_single(X(i, :));
        end
    end
end

function y = predict_single(x)
% PREDICT_SINGLE Predict for a single sample
%
%   Args:
%       x: Single sample as vector of length {len(feature_columns)}
%
%   Returns:
%       y: Predicted value

    % Assign variables
    {"".join([f"{feature} = x({i+1});\n    " for i, feature in enumerate(feature_columns)])}
    
    % Compute prediction
    y = {matlab_expr};
end
"""
        export_formats['matlab'] = matlab_code
        
        # 5. R function
        r_expr = expression
        r_expr = r_expr.replace('np.sin', 'sin')
        r_expr = r_expr.replace('np.cos', 'cos')
        r_expr = r_expr.replace('np.exp', 'exp')
        r_expr = r_expr.replace('np.log', 'log')
        r_expr = r_expr.replace('np.sqrt', 'sqrt')
        r_expr = r_expr.replace('np.power', '^')
        r_expr = r_expr.replace('np.abs', 'abs')
        
        r_code = f"""
#' Make predictions using the discovered symbolic expression.
#'
#' @param X Input features as matrix of shape (n_samples, {len(feature_columns)})
#'          or a single sample of length {len(feature_columns)}
#' @return Predicted values
predict <- function(X) {{
    # Handle both single samples and multiple samples
    if (is.vector(X) && length(X) == {len(feature_columns)}) {{
        # Single sample
        return(predict_single(X))
    }} else {{
        # Multiple samples
        n_samples <- nrow(X)
        y <- numeric(n_samples)
        for (i in 1:n_samples) {{
            y[i] <- predict_single(X[i, ])
        }}
        return(y)
    }}
}}

#' Predict for a single sample
#'
#' @param x Single sample as vector of length {len(feature_columns)}
#' @return Predicted value
predict_single <- function(x) {{
    # Assign variables
    {"".join([f"{feature} <- x[{i+1}]\n    " for i, feature in enumerate(feature_columns)])}
    
    # Compute prediction
    return({r_expr})
}}
"""
        export_formats['r'] = r_code
        
    except Exception as e:
        export_formats['error'] = str(e)
    
    return export_formats


def analyze_feature_importance(expression, feature_columns):
    """
    Analyze feature importance in a symbolic expression
    
    Args:
        expression: Symbolic expression as string
        feature_columns: List of feature column names
        
    Returns:
        importance: Dictionary with feature importance information
    """
    importance = {}
    
    try:
        # Count occurrences of each feature
        for feature in feature_columns:
            # Count direct occurrences
            count = expression.count(feature)
            
            # Store results
            importance[feature] = {
                'count': count,
                'percentage': 0.0  # Will be calculated later
            }
        
        # Calculate percentages
        total_occurrences = sum(importance[feature]['count'] for feature in feature_columns)
        if total_occurrences > 0:
            for feature in feature_columns:
                importance[feature]['percentage'] = 100 * importance[feature]['count'] / total_occurrences
        
        # Analyze interaction terms
        interactions = []
        for i, feat1 in enumerate(feature_columns):
            for feat2 in feature_columns[i+1:]:
                # Look for multiplication or division patterns
                pattern1 = f"{feat1}.*{feat2}"
                pattern2 = f"{feat2}.*{feat1}"
                if re.search(pattern1, expression) or re.search(pattern2, expression):
                    interactions.append((feat1, feat2))
        
        importance['interactions'] = interactions
        
    except Exception as e:
        importance['error'] = str(e)
    
    return importance


def generate_visualizations(model, X, y, X_train, y_train, X_test, y_test, 
                          y_train_pred, y_test_pred, feature_columns):
    """
    Generate visualizations for the XAI dashboard
    
    Args:
        model: Trained model
        X, y: Full dataset
        X_train, y_train, X_test, y_test: Train/test split
        y_train_pred, y_test_pred: Model predictions
        feature_columns: Feature column names
        
    Returns:
        visualizations: Dictionary with visualization data
    """
    visualizations = {}
    
    # Helper function to convert matplotlib figure to base64 image
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    try:
        # 1. Actual vs. Predicted plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_train, y_train_pred, alpha=0.5, label='Training')
        ax.scatter(y_test, y_test_pred, alpha=0.5, label='Test')
        
        # Add perfect prediction line
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs. Predicted')
        ax.legend()
        
        visualizations['actual_vs_predicted'] = fig_to_base64(fig)
        
        # 2. Residual plot
        fig, ax = plt.subplots(figsize=(8, 6))
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        
        ax.scatter(y_train_pred, train_residuals, alpha=0.5, label='Training')
        ax.scatter(y_test_pred, test_residuals, alpha=0.5, label='Test')
        ax.axhline(y=0, color='k', linestyle='--')
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.legend()
        
        visualizations['residuals'] = fig_to_base64(fig)
        
        # 3. Feature effect plots
        feature_effects = {}
        
        # For each feature, create a partial dependence plot
        for i, feature in enumerate(feature_columns):
            if X.shape[1] > i:  # Check if the feature exists
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Sort by feature value
                sorted_indices = np.argsort(X[:, i])
                x_sorted = X[sorted_indices, i]
                y_sorted = y[sorted_indices]
                
                # Plot actual values
                ax.scatter(x_sorted, y_sorted, alpha=0.3, label='Actual')
                
                # Generate model predictions varying only this feature
                if hasattr(model, 'predict') and callable(model.predict):
                    try:
                        # Create feature grid
                        feature_grid = np.linspace(min(x_sorted), max(x_sorted), 100)
                        
                        # Create prediction data - set all features to median except the target feature
                        X_median = np.median(X, axis=0)
                        X_pred = np.tile(X_median, (100, 1))
                        X_pred[:, i] = feature_grid
                        
                        # Get predictions
                        y_pred = model.predict(X_pred)
                        
                        # Plot prediction curve
                        ax.plot(feature_grid, y_pred, 'r-', label='Model Prediction')
                    except:
                        pass
                
                ax.set_xlabel(feature)
                ax.set_ylabel('Target')
                ax.set_title(f'Effect of {feature} on Target')
                ax.legend()
                
                feature_effects[feature] = fig_to_base64(fig)
        
        visualizations['feature_effects'] = feature_effects
        
        # 4. Expression visualizations
        if hasattr(model, 'get_best_expression'):
            expression = model.get_best_expression()
            
            try:
                # Parse expression using sympy
                expr = sp.sympify(expression)
                
                # Convert to LaTeX for better rendering
                latex_expr = sp.latex(expr)
                visualizations['expression_latex'] = latex_expr
                
                # Create expression tree visualization if expression is not too complex
                if len(expression) < 500:  # Limit for very complex expressions
                    # This is a simple tree representation using matplotlib
                    def build_expr_tree(expr, depth=0, x_pos=0, ax=None):
                        if ax is None:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.set_xlim(-1, 1)
                            ax.set_ylim(0, 1)
                            ax.axis('off')
                        
                        # Draw node
                        node_str = str(expr.func) if hasattr(expr, 'func') else str(expr)
                        if len(node_str) > 20:
                            node_str = node_str[:17] + "..."
                        
                        # Position based on depth
                        y_pos = 1 - depth * 0.1
                        
                        # Draw node
                        ax.text(x_pos, y_pos, node_str, 
                              ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
                        
                        # Process children
                        if hasattr(expr, 'args') and expr.args:
                            n_children = len(expr.args)
                            width = 0.8 / (2 ** depth)
                            
                            for i, arg in enumerate(expr.args):
                                # Calculate child position
                                child_x = x_pos + width * (i - n_children / 2 + 0.5)
                                
                                # Draw edge
                                ax.plot([x_pos, child_x], [y_pos - 0.02, y_pos - 0.08], 'k-')
                                
                                # Recursively draw child
                                build_expr_tree(arg, depth + 1, child_x, ax)
                        
                        return ax
                    
                    # Create the tree visualization
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax = build_expr_tree(expr, ax=ax)
                    ax.set_title("Expression Tree Visualization")
                    
                    visualizations['expression_tree'] = fig_to_base64(fig)
            except:
                pass
        
        # 5. Additional model-specific visualizations
        if hasattr(model, 'plot_pareto_front'):
            try:
                fig, ax = model.plot_pareto_front()
                visualizations['pareto_front'] = fig_to_base64(fig)
            except:
                pass
        
        if hasattr(model, 'plot_convergence'):
            try:
                fig, ax = model.plot_convergence()
                visualizations['convergence'] = fig_to_base64(fig)
            except:
                pass
        
        if hasattr(model, 'plot_complexity_progression'):
            try:
                fig, ax = model.plot_complexity_progression()
                visualizations['complexity_progression'] = fig_to_base64(fig)
            except:
                pass
    
    except Exception as e:
        visualizations['error'] = str(e)
    
    return visualizations


# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_data():
    """Handle data upload"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    data, message = parse_uploaded_data(file)
    
    if data is None:
        return jsonify({'success': False, 'message': message})
    
    # Generate a session ID if not exists
    if 'id' not in session:
        session['id'] = str(uuid.uuid4())
    
    # Save data to session
    session_id = session['id']
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_data.pkl")
    
    with open(session_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Return columns and preview
    columns = data.columns.tolist()
    preview = data.head(5).to_dict(orient='records')
    
    return jsonify({
        'success': True, 
        'message': message,
        'columns': columns,
        'preview': preview,
        'rows': len(data)
    })


@app.route('/start_training', methods=['POST'])
def start_training():
    """Start a symbolic regression training task"""
    # Check if data is available
    if 'id' not in session:
        return jsonify({'success': False, 'message': 'No data available. Please upload data first.'})
    
    session_id = session['id']
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_data.pkl")
    
    if not os.path.exists(session_file):
        return jsonify({'success': False, 'message': 'Data not found. Please upload data again.'})
    
    # Load data
    with open(session_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get parameters from request
    params = request.json
    
    # Validate parameters
    if 'target_column' not in params:
        return jsonify({'success': False, 'message': 'Target column not specified'})
    
    if 'feature_columns' not in params or not params['feature_columns']:
        return jsonify({'success': False, 'message': 'No feature columns selected'})
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Store task info
    active_tasks[task_id] = {
        'id': task_id,
        'status': 'queued',
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': params,
        'progress': 0,
        'current_metrics': {}
    })
    
    # Add task to queue
    task_queue.put((
        task_id, 
        train_symbolic_regression, 
        (data, params['target_column'], params['feature_columns'], params, task_id), 
        {}
    ))
    
    return jsonify({
        'success': True, 
        'message': 'Training started',
        'task_id': task_id
    })


@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the status of a task"""
    if task_id not in active_tasks:
        return jsonify({'success': False, 'message': 'Task not found'})
    
    task_info = active_tasks[task_id].copy()
    
    # Check if task is completed
    if task_id in task_results:
        task_info['status'] = 'completed'
        
        # Get basic result info without the model
        result_info = {
            'algorithm': task_results[task_id]['algorithm'],
            'expression': task_results[task_id]['expression'],
            'simplified_expression': task_results[task_id]['simplified_expression'],
            'train_mse': task_results[task_id]['train_mse'],
            'test_mse': task_results[task_id]['test_mse'],
            'train_r2': task_results[task_id]['train_r2'],
            'test_r2': task_results[task_id]['test_r2'],
            'execution_time': task_results[task_id]['execution_time']
        }
        
        task_info['result'] = result_info
    
    return jsonify({
        'success': True,
        'task': task_info
    })


@app.route('/task_result/<task_id>', methods=['GET'])
def get_task_result(task_id):
    """Get the full result of a completed task"""
    if task_id not in task_results:
        return jsonify({'success': False, 'message': 'Task result not found'})
    
    result = task_results[task_id]
    
    # Don't send the model object in the response
    result_copy = result.copy()
    if 'model' in result_copy:
        del result_copy['model']
    if 'feature_scaler' in result_copy:
        del result_copy['feature_scaler']
    if 'target_scaler' in result_copy:
        del result_copy['target_scaler']
    
    return jsonify({
        'success': True,
        'result': result_copy
    })


@app.route('/dashboard/<task_id>', methods=['GET'])
def xai_dashboard(task_id):
    """Render the XAI dashboard for a completed task"""
    if task_id not in task_results:
        return render_template('error.html', message='Task result not found')
    
    # Pass the task ID to the template
    return render_template('dashboard.html', task_id=task_id)


@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    """Make predictions using a trained model"""
    # Get parameters
    params = request.json
    
    if 'task_id' not in params:
        return jsonify({'success': False, 'message': 'Task ID not specified'})
    
    if 'values' not in params:
        return jsonify({'success': False, 'message': 'No values provided'})
    
    task_id = params['task_id']
    values = params['values']
    
    if task_id not in task_results:
        return jsonify({'success': False, 'message': 'Task result not found'})
    
    result = task_results[task_id]
    
    if 'model' not in result:
        return jsonify({'success': False, 'message': 'Model not found in task result'})
    
    model = result['model']
    feature_columns = result['feature_columns']
    feature_scaler = result['feature_scaler']
    target_scaler = result['target_scaler']
    
    # Validate input values
    if len(values) != len(feature_columns):
        return jsonify({'success': False, 'message': f'Expected {len(feature_columns)} values, got {len(values)}'})
    
    try:
        # Convert values to float
        input_values = [float(v) for v in values]
        
        # Create input array
        X_pred = np.array([input_values])
        
        # Apply scaling if used
        if feature_scaler is not None:
            X_pred = feature_scaler.transform(X_pred)
        
        # Make prediction
        y_pred = model.predict(X_pred)
        
        # Inverse transform if scaled
        if target_scaler is not None:
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': float(y_pred[0]),
            'expression': result['expression'],
            'simplified_expression': result['simplified_expression']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error making prediction: {str(e)}'})


@app.route('/explain_prediction', methods=['POST'])
def explain_prediction():
    """Explain a specific prediction"""
    # Get parameters
    params = request.json
    
    if 'task_id' not in params:
        return jsonify({'success': False, 'message': 'Task ID not specified'})
    
    if 'values' not in params:
        return jsonify({'success': False, 'message': 'No values provided'})
    
    task_id = params['task_id']
    values = params['values']
    
    if task_id not in task_results:
        return jsonify({'success': False, 'message': 'Task result not found'})
    
    result = task_results[task_id]
    expression = result['simplified_expression']
    feature_columns = result['feature_columns']
    
    # Validate input values
    if len(values) != len(feature_columns):
        return jsonify({'success': False, 'message': f'Expected {len(feature_columns)} values, got {len(values)}'})
    
    try:
        # Convert values to float
        input_values = [float(v) for v in values]
        
        # Parse expression
        expr_str = expression
        for i, feature in enumerate(feature_columns):
            expr_str = expr_str.replace(feature, f"({input_values[i]})")
        
        # Evaluate expression
        prediction = eval(expr_str)
        
        # Create explanation
        steps = []
        
        # Parse the expression into its component parts
        try:
            sympy_expr = sp.sympify(expression)
            
            # Function to recursively explain expression
            def explain_expression(expr, values_dict):
                if isinstance(expr, sp.Symbol):
                    # It's a variable
                    name = expr.name
                    if name in values_dict:
                        return f"{name} = {values_dict[name]}"
                    return name
                elif isinstance(expr, sp.Number):
                    # It's a constant
                    return str(expr)
                elif len(expr.args) > 0:
                    # It's an operation
                    op_name = str(expr.func)
                    op_name = op_name.replace("numpy.", "").replace("<function ", "").replace(" at 0x[0-9a-f]+>", "")
                    
                    # Explain each argument
                    arg_explanations = [explain_expression(arg, values_dict) for arg in expr.args]
                    
                    # Calculate result of this operation
                    result = expr.subs(values_dict)
                    
                    # Generate explanation based on operation type
                    if isinstance(expr, sp.Add):
                        explanation = f"({' + '.join(arg_explanations)}) = {result}"
                    elif isinstance(expr, sp.Mul):
                        explanation = f"({' × '.join(arg_explanations)}) = {result}"
                    elif isinstance(expr, sp.Pow):
                        explanation = f"({arg_explanations[0]})^({arg_explanations[1]}) = {result}"
                    elif op_name in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                        explanation = f"{op_name}({arg_explanations[0]}) = {result}"
                    else:
                        explanation = f"{op_name}({', '.join(arg_explanations)}) = {result}"
                    
                    steps.append(explanation)
                    return str(result)
            
            # Create values dictionary
            values_dict = {sp.Symbol(feature): value for feature, value in zip(feature_columns, input_values)}
            
            # Generate explanation
            explain_expression(sympy_expr, values_dict)
            
        except Exception as e:
            # Fallback to simple explanation
            steps.append(f"Substituting values: {expr_str}")
            steps.append(f"Result: {prediction}")
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'expression': expression,
            'steps': steps
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error explaining prediction: {str(e)}'})


@app.route('/export_model/<task_id>', methods=['GET'])
def export_model(task_id):
    """Export the trained model to Python code format"""
    if task_id not in task_results:
        return jsonify({'success': False, 'message': 'Task result not found'})
    
    result = task_results[task_id]
    
    if 'export_formats' not in result or 'python' not in result['export_formats']:
        return jsonify({'success': False, 'message': 'Export format not available'})
    
    python_code = result['export_formats']['python']
    
    # Create a downloadable file
    response = Response(
        python_code,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename=symbolic_regression_model.py"}
    )
    
    return response


@app.route('/export_model/<task_id>/<format>', methods=['GET'])
def export_model_format(task_id, format):
    """Export the trained model in a specific format"""
    if task_id not in task_results:
        return jsonify({'success': False, 'message': 'Task result not found'})
    
    result = task_results[task_id]
    
    if 'export_formats' not in result or format not in result['export_formats']:
        return jsonify({'success': False, 'message': f'Export format {format} not available'})
    
    code = result['export_formats'][format]
    
    # File extension depends on format
    extensions = {
        'python': '.py',
        'cpp': '.cpp',
        'matlab': '.m',
        'r': '.R',
        'latex': '.tex'
    }
    
    filename = f"symbolic_regression_model{extensions.get(format, '.txt')}"
    
    # Create a downloadable file
    response = Response(
        code,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )
    
    return response


# Create the HTML templates
@app.route('/templates/init', methods=['GET'])
def initialize_templates():
    """Initialize the HTML templates"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    with open(os.path.join(templates_dir, 'base.html'), 'w') as f:
        f.write(base_html)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
        f.write(dashboard_html)
    
    with open(os.path.join(templates_dir, 'error.html'), 'w') as f:
        f.write(error_html)
    
    return "Templates initialized successfully"


if __name__ == '__main__':
    # Initialize templates
    app.route('/templates/init')(initialize_templates)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
