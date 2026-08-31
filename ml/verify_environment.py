#!/usr/bin/env python
"""
Environment Verification Script

This script verifies that all required ML pipeline packages are properly installed
and displays their versions. Run this after installing requirements.txt to ensure
your environment is set up correctly.

Usage:
    python verify_environment.py
"""

import sys
from pathlib import Path


def verify_package(package_name, import_name=None):
    """
    Verify a single package is installed and return its version.
    
    Args:
        package_name: Name of the package (for display)
        import_name: Name to use in import statement (if different from package_name)
    
    Returns:
        Tuple (success: bool, version: str or error message)
    """
    import_name = import_name or package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown version')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    """Main verification function."""
    
    print("=" * 70)
    print("SmartStock AI - ML Pipeline Environment Verification")
    print("=" * 70)
    print()
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()
    
    # List of packages to verify
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('xgboost', 'xgboost'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('jupyter', 'jupyter'),
        ('pytest', 'pytest'),
        ('python-dotenv', 'dotenv'),
    ]
    
    print("Verifying required packages:")
    print("-" * 70)
    
    all_success = True
    results = []
    
    for package_name, import_name in packages:
        success, version = verify_package(package_name, import_name)
        results.append((package_name, success, version))
        
        if success:
            status = "✓ OK"
            all_success = all_success and True
        else:
            status = "✗ FAILED"
            all_success = False
        
        print(f"{package_name:20} {status:10} {version}")
    
    print("-" * 70)
    print()
    
    if all_success:
        print("✓ SUCCESS: All packages are correctly installed!")
        print()
        print("Your ML pipeline environment is ready for development.")
        print("You can now:")
        print("  - Import modules in Python scripts")
        print("  - Run Jupyter notebooks: jupyter notebook")
        print("  - Run tests: pytest tests/ml/")
        print()
        return 0
    else:
        print("✗ FAILURE: Some packages are missing or incorrectly installed.")
        print()
        print("Failed packages:")
        for package_name, success, error in results:
            if not success:
                print(f"  - {package_name}: {error}")
        print()
        print("To fix this, run:")
        print("  pip install -r requirements.txt")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
