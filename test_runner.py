#!/usr/bin/env python3
"""
Test Runner for AI Kubernetes Scheduler
Automates testing of pod scheduling with the AI scheduler
"""

import subprocess
import time
import sys
from kubernetes import client, config

def run_command(cmd, check=True):
    """Run a shell command and return output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def main():
    print("🧪 AI Scheduler Test Runner")
    print("=" * 60)
    
    # Load kubeconfig
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        print("✅ Connected to Kubernetes cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Kubernetes: {e}")
        sys.exit(1)
    
    # Clean up any existing test pods
    print("\n🧹 Cleaning up existing test pods...")
    run_command("kubectl delete -f ai-test-pods.yaml --ignore-not-found=true", check=False)
    time.sleep(2)
    
    # Create test pods
    print("\n📦 Creating test pods...")
    run_command("kubectl apply -f ai-test-pods.yaml")
    
    # Wait for scheduling
    print("\n⏳ Waiting for pods to be scheduled (30 seconds)...")
    time.sleep(30)
    
    # Check pod status
    print("\n📊 Pod Status:")
    print("-" * 60)
    
    pods = v1.list_namespaced_pod(namespace="default")
    test_pods = [p for p in pods.items if p.metadata.name.startswith("ai-test-pod")]
    
    scheduled_count = 0
    for pod in test_pods:
        status = pod.status.phase
        node = pod.spec.node_name or "Not scheduled"
        
        if pod.spec.node_name:
            scheduled_count += 1
            status_icon = "✅"
        else:
            status_icon = "⏳"
        
        print(f"{status_icon} {pod.metadata.name:20} | Status: {status:10} | Node: {node}")
    
    print("-" * 60)
    print(f"\n📈 Results: {scheduled_count}/{len(test_pods)} pods scheduled")
    
    if scheduled_count == len(test_pods):
        print("✅ All pods successfully scheduled!")
        return 0
    else:
        print("⚠️  Some pods not scheduled yet. Check scheduler logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
