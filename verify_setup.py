#!/usr/bin/env python3
"""
Quick Setup Verification Script
Checks if all dependencies and configuration are ready
"""

import sys
import os
from pathlib import Path

def check_file(filepath, required=True):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "" if exists else ("" if required else "")
    req_text = "(required)" if required else "(optional)"
    print(f"{status} {filepath} {req_text}")
    return exists

def check_env_var(var_name, required=True):
    """Check if environment variable is set"""
    value = os.getenv(var_name)
    exists = value is not None and value != ""
    status = "" if exists else ("" if required else "")
    req_text = "(required)" if required else "(optional)"
    print(f"{status} {var_name} {req_text}")
    return exists

def main():
    print(" AI Kubernetes Scheduler - Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check files
    print("\n Checking files...")
    files_ok = all([
        check_file("scheduler.py"),
        check_file("config.yaml"),
        check_file("requirements.txt"),
        check_file("ai-test-pods.yaml"),
        check_file(".env", required=False),
        check_file(".env.example"),
    ])
    
    # Check environment variables
    print("\n Checking environment variables...")
    
    # Try to load .env if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  python-dotenv not installed yet (run: pip install -r requirements.txt)")
    
    env_ok = check_env_var("HUGGINGFACE_TOKEN")
    check_env_var("SCHEDULER_NAME", required=False)
    check_env_var("LLM_MODEL", required=False)
    
    # Check Python packages
    print("\n Checking Python packages...")
    packages_check = [
        ("kubernetes", "kubernetes"),
        ("huggingface_hub", "huggingface_hub"),
        ("yaml", "yaml"),
        ("dotenv", "dotenv")
    ]
    
    packages_ok = True
    for display_name, import_name in packages_check:
        try:
            __import__(import_name)
            print(f" {display_name}")
        except ImportError:
            print(f" {display_name} (run: pip install -r requirements.txt)")
            packages_ok = False
    
    # Check Kubernetes connection
    print("\n  Checking Kubernetes connection...")
    try:
        from kubernetes import client, config
        config.load_kube_config()
        v1 = client.CoreV1Api()
        nodes = v1.list_node()
        print(f" Connected to Kubernetes ({len(nodes.items)} nodes)")
        k8s_ok = True
    except Exception as e:
        print(f" Cannot connect to Kubernetes: {e}")
        print("   Run: minikube start --nodes 3")
        k8s_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print(" Summary:")
    print("=" * 60)
    
    all_good = files_ok and env_ok and packages_ok and k8s_ok
    
    if all_good:
        print(" All checks passed! You're ready to run the scheduler.")
        print("\nNext steps:")
        print("  1. python scheduler.py")
        print("  2. kubectl apply -f ai-test-pods.yaml")
        return 0
    else:
        print(" Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        if not packages_ok:
            print("  • pip install -r requirements.txt")
        if not env_ok:
            print("  • cp .env.example .env")
            print("  • Edit .env and add your HUGGINGFACE_TOKEN")
        if not k8s_ok:
            print("  • minikube start --nodes 3")
        return 1

if __name__ == "__main__":
    sys.exit(main())
