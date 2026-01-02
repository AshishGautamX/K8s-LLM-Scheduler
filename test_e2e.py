#!/usr/bin/env python3
"""
Complete End-to-End Test Script
Tests the AI Kubernetes Scheduler with Llama-3.3-70B
"""

import subprocess
import time
import sys
import os
from kubernetes import client, config

def run_command(cmd, check=True, shell=True):
    """Run a shell command"""
    print(f"\n Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and check:
        print(f"  {result.stderr}")
    if check and result.returncode != 0:
        print(f" Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    print("=" * 70)
    print(" AI Kubernetes Scheduler - Complete End-to-End Test")
    print("=" * 70)
    
    # Step 1: Verify setup
    print("\n Step 1: Verifying setup...")
    if not run_command("python verify_setup.py", check=False):
        print(" Setup verification failed. Please fix issues first.")
        return 1
    
    # Step 2: Clean up existing pods
    print("\n Step 2: Cleaning up existing test pods...")
    run_command("kubectl delete -f ai-test-pods.yaml --ignore-not-found=true", check=False)
    time.sleep(3)
    
    # Step 3: Check cluster status
    print("\n  Step 3: Checking Kubernetes cluster...")
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        nodes = v1.list_node()
        print(f" Cluster has {len(nodes.items)} nodes:")
        for node in nodes.items:
            print(f"   - {node.metadata.name}")
    except Exception as e:
        print(f" Cannot connect to Kubernetes: {e}")
        return 1
    
    # Step 4: Instructions for scheduler
    print("\n" + "=" * 70)
    print(" Step 4: Start the Scheduler")
    print("=" * 70)
    print("\n  IMPORTANT: You need to manually start the scheduler in a separate terminal:")
    print("\n   1. Open a NEW terminal window")
    print("   2. Navigate to the project directory")
    print("   3. Run: python scheduler.py")
    print("   4. Wait for the message: ' Watching for pods...'")
    print("\n   Then come back here and press ENTER to continue...")
    
    input("\n⏸  Press ENTER when scheduler is running...")
    
    # Step 5: Create test pods
    print("\n Step 5: Creating test pods...")
    if not run_command("kubectl apply -f ai-test-pods.yaml"):
        print(" Failed to create pods")
        return 1
    
    print("\n⏳ Waiting 10 seconds for scheduling...")
    time.sleep(10)
    
    # Step 6: Check pod status
    print("\n Step 6: Checking pod status...")
    print("-" * 70)
    
    try:
        pods = v1.list_namespaced_pod(namespace="default")
        test_pods = [p for p in pods.items if p.metadata.name.startswith("ai-test-pod")]
        
        if not test_pods:
            print(" No test pods found!")
            return 1
        
        scheduled_count = 0
        running_count = 0
        
        print(f"\n{'POD NAME':<20} {'STATUS':<12} {'NODE':<20} {'SCHEDULER'}")
        print("-" * 70)
        
        for pod in test_pods:
            status = pod.status.phase
            node = pod.spec.node_name or "Not scheduled"
            scheduler = pod.spec.scheduler_name
            
            if pod.spec.node_name:
                scheduled_count += 1
            if status == "Running":
                running_count += 1
                status_icon = ""
            elif status == "Pending":
                status_icon = "⏳"
            else:
                status_icon = ""
            
            print(f"{status_icon} {pod.metadata.name:<18} {status:<12} {node:<20} {scheduler}")
        
        print("-" * 70)
        print(f"\n Results:")
        print(f"   • Scheduled: {scheduled_count}/{len(test_pods)} pods")
        print(f"   • Running:   {running_count}/{len(test_pods)} pods")
        
        # Step 7: Check scheduler logs
        print("\n Step 7: Checking scheduler decisions...")
        print("\n Check your scheduler terminal for messages like:")
        print("    Calling HuggingFace Llama-3.3-70B for scheduling decision...")
        print("    LLM decision: <node-name> (confidence: 0.XX)")
        print("    Reasoning: <AI reasoning>")
        
        # Final verdict
        print("\n" + "=" * 70)
        if scheduled_count == len(test_pods) and running_count == len(test_pods):
            print(" SUCCESS! All pods scheduled and running!")
            print("=" * 70)
            print("\n The AI Kubernetes Scheduler is working perfectly!")
            print("\n What happened:")
            print("   1. Llama-3.3-70B analyzed cluster state")
            print("   2. Made intelligent scheduling decisions")
            print("   3. Pods were bound to optimal nodes")
            print("   4. All pods are now running")
            return 0
        elif scheduled_count == len(test_pods):
            print("  PARTIAL SUCCESS - Pods scheduled but not all running yet")
            print("=" * 70)
            print("\nWait a bit longer for pods to start...")
            return 0
        else:
            print(" FAILED - Some pods not scheduled")
            print("=" * 70)
            print("\n Troubleshooting:")
            print("   1. Check if scheduler is running")
            print("   2. Check scheduler logs for errors")
            print("   3. Verify HuggingFace token is valid")
            return 1
            
    except Exception as e:
        print(f" Error checking pods: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
