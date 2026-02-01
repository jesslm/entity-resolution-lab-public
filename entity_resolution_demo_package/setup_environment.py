#!/usr/bin/env python3
"""
Environment setup and verification script for the Entity Resolution Demo package.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def _repo_root() -> Path:
    """
    Return the repo root directory regardless of current working directory.

    This file lives at: <repo_root>/entity_resolution_demo_package/setup_environment.py
    """
    return Path(__file__).resolve().parent.parent

def check_virtual_environment():
    """Warn (but do not fail) if user is not running inside a virtual environment."""
    print("\n🧪 Checking virtual environment...")

    in_venv = (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
        or hasattr(sys, "real_prefix")
        or bool(os.environ.get("VIRTUAL_ENV"))
        or bool(os.environ.get("CONDA_PREFIX"))
    )

    if in_venv:
        print("✅ Virtual environment detected")
        return True

    print("⚠️  No virtual environment detected.")
    print("💡 Recommended: create one (venv/conda) before installing dependencies.")
    print("   Example (venv): python -m venv .venv && source .venv/bin/activate")
    return True

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_package_installation():
    """Check if the package is properly installed."""
    print("\n📦 Checking package installation...")
    try:
        import entity_resolution_demo
        print("✅ Package is installed")
        return True
    except ImportError as e:
        print(f"❌ Package not found: {e}")
        print("💡 Run: pip install -e .")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔧 Checking dependencies...")
    # (pip_distribution_name, python_import_name)
    required_packages = [
        ("elasticsearch", "elasticsearch"),
        ("openai", "openai"),
        ("requests", "requests"),
        ("python-dotenv", "dotenv"),
        ("pydantic", "pydantic"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]
    
    missing_packages = []
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    return True

def check_tutorial_dependencies():
    """Check if tutorial dependencies are installed."""
    print("\n📚 Checking tutorial dependencies...")
    tutorial_packages = [
        "jupyter",
        "matplotlib", 
        "seaborn",
        "ipywidgets"
    ]
    
    missing_tutorial = []
    for package in tutorial_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_tutorial.append(package)
    
    if missing_tutorial:
        print(f"\n💡 Install tutorial dependencies: pip install -e \".[tutorial]\"")
        return False
    return True

def check_environment_file():
    """Check if .env file exists and is configured."""
    print("\n🔐 Checking environment configuration...")
    
    # Load .env file if it exists
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        print("❌ .env file not found")
        print("💡 Copy env.template to .env and configure your credentials")
        return False
    
    load_dotenv(dotenv_path=env_file)
    
    elastic_endpoint = os.getenv("ELASTIC_ENDPOINT")
    elastic_cloud_id = os.getenv("ELASTIC_CLOUD_ID")

    required_vars = ["ELASTIC_API_KEY", "OPENAI_API_KEY"]
    
    missing_vars = []
    if elastic_endpoint:
        print("✅ ELASTIC_ENDPOINT is configured")
    elif elastic_cloud_id:
        print("✅ ELASTIC_CLOUD_ID is configured")
    else:
        print("❌ Neither ELASTIC_ENDPOINT nor ELASTIC_CLOUD_ID is set")
        missing_vars.append("ELASTIC_ENDPOINT/ELASTIC_CLOUD_ID")

    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is configured")
        else:
            print(f"❌ {var} not set")
            missing_vars.append(var)
    
    if missing_vars:
        print("\n💡 Set these environment variables in your .env file:")
        print("- ELASTIC_API_KEY")
        print("- Either ELASTIC_ENDPOINT (preferred) or ELASTIC_CLOUD_ID")
        print("- OPENAI_API_KEY")
        return False
    return True

def check_data_files():
    """Check if required data files exist."""
    print("\n📄 Checking data files...")
    
    repo_root = _repo_root()
    data_files = [
        "notebooks/minimal_entities.json",
        "notebooks/minimal_articles.json"
    ]
    
    missing_files = []
    for file_path in data_files:
        full_path = repo_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n💡 Ensure data files are present in the notebooks directory")
        return False
    return True

def test_elasticsearch_connection():
    """Test Elasticsearch connection."""
    print("\n🔍 Testing Elasticsearch connection...")
    try:
        from entity_resolution_demo.search.elastic_client import ElasticClient
        client = ElasticClient(allow_local_fallback=False)
        if client.check_connection():
            print("✅ Elasticsearch connection successful")
            return True
        else:
            print("❌ Elasticsearch connection failed")
            return False
    except Exception as e:
        print(f"❌ Elasticsearch connection error: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🤖 Testing OpenAI API connection...")
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        print("✅ OpenAI API connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI API connection error: {e}")
        return False

def main():
    """Run all environment checks."""
    print("🚀 Entity Resolution Demo - Environment Setup")
    print("=" * 50)
    
    checks = [
        ("Virtual Environment (Recommended)", check_virtual_environment),
        ("Python Version", check_python_version),
        ("Package Installation", check_package_installation),
        ("Dependencies", check_dependencies),
        ("Tutorial Dependencies", check_tutorial_dependencies),
        ("Environment Configuration", check_environment_file),
        ("Data Files", check_data_files),
        ("Elasticsearch Connection", test_elasticsearch_connection),
        ("OpenAI Connection", test_openai_connection),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start learning!")
        print("\n📚 Next steps:")
        repo_root = _repo_root()
        print("1. Start Jupyter from the repo root (important):")
        print(f"   cd {repo_root}")
        print("   jupyter lab")
        print("2. Open: notebooks/01_entity_preparation_v3.ipynb")
        print("3. Follow the educational scenarios")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\n💡 Common solutions:")
        print("- Install package: pip install -e .")
        print("- Install tutorial deps: pip install -e \".[tutorial]\"")
        print("- Configure .env file with your credentials")
        print("- Check Elasticsearch Cloud and OpenAI API access")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
