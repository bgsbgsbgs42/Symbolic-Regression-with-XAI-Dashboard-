"""
Launcher script for the Symbolic Regression Web App with XAI Dashboard.

This script sets up the environment and launches the web application.
It supports both local development and cloud deployment.
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path
import json

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask', 'numpy', 'pandas', 'scikit-learn', 'matplotlib',
        'sympy', 'deap', 'gunicorn', 'waitress'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    print(f"Installing missing dependencies: {', '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def initialize_directory_structure():
    """Create necessary directories"""
    # Create directories
    dirs = ['uploads', 'templates', 'static']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Directory structure initialized.")

def launch_app_development(port=5000, browser=True):
    """Launch the web application in development mode"""
    from symbolic_regression_webapp import app
    import threading
    
    # Initialize templates
    response = app.test_client().get('/templates/init')
    if response.status_code != 200:
        print("Warning: Failed to initialize HTML templates.")
    
    # Open browser if requested
    if browser:
        url = f"http://localhost:{port}"
        # Give Flask a moment to start
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    # Run Flask app in development mode
    print(f"Starting Symbolic Regression Web App on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port)

def launch_app_production(port=5000, workers=4):
    """Launch the web application in production mode"""
    print(f"Starting Symbolic Regression Web App in production mode on port {port}...")
    
    # Determine which production server is available
    try:
        import gunicorn
        # We're on Linux/Mac, use gunicorn
        cmd = [
            'gunicorn',
            '--bind', f'0.0.0.0:{port}',
            '--workers', str(workers),
            '--timeout', '120',
            'symbolic_regression_webapp:app'
        ]
        subprocess.run(cmd)
    except ImportError:
        try:
            import waitress
            # We're on Windows, use waitress
            from waitress import serve
            from symbolic_regression_webapp import app
            print(f"Using waitress to serve the application on port {port}...")
            serve(app, host='0.0.0.0', port=port, threads=workers)
        except ImportError:
            print("Neither gunicorn nor waitress is installed. Falling back to Flask development server.")
            launch_app_development(port=port, browser=False)

def create_deployment_files():
    """Create files needed for cloud deployment"""
    # Create Procfile for Heroku
    with open('Procfile', 'w') as f:
        f.write('web: gunicorn symbolic_regression_webapp:app')
    
    # Create requirements.txt
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=open('requirements.txt', 'w'))
    
    # Create runtime.txt
    with open('runtime.txt', 'w') as f:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        f.write(f"python-{python_version}")
    
    # Create app.yaml for Google App Engine
    with open('app.yaml', 'w') as f:
        f.write(f"""runtime: python{sys.version_info.major}{sys.version_info.minor}
instance_class: F2
entrypoint: gunicorn -b :$PORT symbolic_regression_webapp:app

handlers:
- url: /.*
  script: auto
""")
    
    # Create .dockerignore
    with open('.dockerignore', 'w') as f:
        f.write("""
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
env/
venv/
ENV/
.idea/
.git/
.gitignore
uploads/
""")
    
    # Create Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(f"""FROM python:{sys.version_info.major}.{sys.version_info.minor}-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p uploads templates static

# Initialize templates
RUN python -c "from symbolic_regression_webapp import app, initialize_templates; initialize_templates()"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Start the application
CMD gunicorn --bind 0.0.0.0:$PORT symbolic_regression_webapp:app
""")
    
    print("Deployment files created successfully!")


def main():
    """Main function for launching the app"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch Symbolic Regression Web App')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies without launching')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for production mode')
    parser.add_argument('--prepare-deploy', action='store_true', help='Prepare files for deployment')
    
    args = parser.parse_args()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"Missing dependencies: {', '.join(missing_packages)}")
        install = input("Do you want to install them now? (y/n): ")
        
        if install.lower() == 'y':
            install_dependencies(missing_packages)
        else:
            print("Cannot continue without required dependencies.")
            sys.exit(1)
    
    # Exit if check-only flag is set
    if args.check_only:
        print("All dependencies are installed.")
        sys.exit(0)
    
    # Create deployment files if requested
    if args.prepare_deploy:
        create_deployment_files()
        sys.exit(0)
    
    # Initialize directory structure
    initialize_directory_structure()
    
    # Launch the app
    try:
        if args.production:
            # Production mode
            launch_app_production(port=args.port, workers=args.workers)
        else:
            # Development mode
            launch_app_development(port=args.port, browser=not args.no_browser)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()