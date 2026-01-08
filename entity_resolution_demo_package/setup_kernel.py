#!/usr/bin/env python3
"""
Setup script to create the Jupyter kernel for Entity Resolution Demo.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_kernel():
    """Install the Jupyter kernel for Entity Resolution Demo."""
    print("🔧 Setting up Jupyter kernel for Entity Resolution Demo...")
    
    try:
        # Install ipykernel if not already installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])
        
        # Install the kernel
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", 
            "--user", 
            "--name=entity-resolution", 
            "--display-name=Entity Resolution Demo"
        ])
        
        print("✅ Kernel installed successfully!")
        print("📚 You can now start Jupyter and select 'Entity Resolution Demo' as the kernel")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install kernel: {e}")
        return False

def verify_kernel():
    """Verify that the kernel is installed."""
    print("\n🔍 Verifying kernel installation...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "jupyter", "kernelspec", "list"], 
                              capture_output=True, text=True)
        
        if "entity-resolution" in result.stdout:
            print("✅ Kernel verification successful!")
            return True
        else:
            print("❌ Kernel not found in available kernels")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to verify kernel: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Entity Resolution Demo - Kernel Setup")
    print("=" * 50)
    
    # Install the kernel
    if install_kernel():
        # Verify installation
        if verify_kernel():
            print("\n🎉 Kernel setup complete!")
            print("\n📚 Next steps:")
            print("1. Start Jupyter: jupyter lab")
            print("2. Open a notebook")
            print("3. Select 'Entity Resolution Demo' as the kernel")
            print("4. Run the notebooks!")
        else:
            print("\n⚠️  Kernel installation may have failed")
    else:
        print("\n❌ Kernel setup failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
