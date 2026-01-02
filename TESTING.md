#  Complete Testing Guide

## Current Status
-  All dependencies installed
-  Environment configured (HuggingFace token set)
-  Kubernetes cluster running (3 nodes)
-  Code fixed (timeout parameter removed)
-  Old test pods cleaned up

## How to Run Complete Test

### Option 1: Automated Test (Recommended)

1. **Stop the current scheduler** (if running):
   - Go to the terminal running `python scheduler.py`
   - Press `Ctrl+C`

2. **Run the automated test**:
   ```bash
   python test_e2e.py
   ```
   
   This will:
   - Verify all setup
   - Clean up old pods
   - Guide you to start the scheduler
   - Create test pods
   - Verify scheduling worked
   - Show results

### Option 2: Manual Test

1. **Terminal 1 - Start Scheduler**:
   ```bash
   python scheduler.py
   ```
   
   Wait for:
   ```
    Scheduler initialized successfully!
    Watching for pods with schedulerName=ai-llama-scheduler
   ```

2. **Terminal 2 - Create Test Pods**:
   ```bash
   # Clean up first
   kubectl delete -f ai-test-pods.yaml --ignore-not-found=true
   
   # Create new pods
   kubectl apply -f ai-test-pods.yaml
   ```

3. **Watch the scheduler logs** in Terminal 1 for:
   ```
    Detected pod: default/ai-test-pod-X
    Calling HuggingFace Llama-3.3-70B for scheduling decision...
    LLM decision: minikube-mXX (confidence: 0.XX)
    Reasoning: <AI's reasoning>
    Bound pod default/ai-test-pod-X to node minikube-mXX
   ```

4. **Check pod status**:
   ```bash
   kubectl get pods -o wide
   ```

## What to Look For

###  Success Indicators:
- Scheduler logs show: " Calling HuggingFace Llama-3.3-70B..."
- No error messages about timeout
- Pods show "Running" status
- Pods are distributed across nodes
- LLM provides reasoning for decisions

###  Failure Indicators:
- "Using fallback" messages (means LLM call failed)
- Timeout errors
- Pods stuck in "Pending"
- Connection errors to HuggingFace

## Expected Output

### Scheduler Terminal:
```
 AI-Powered Kubernetes Scheduler with Llama-3.3-70B
============================================================
Using HuggingFace Inference API
Model: meta-llama/Llama-3.3-70B-Instruct
============================================================
 Initialized HuggingFace client for model: meta-llama/Llama-3.3-70B-Instruct
 Endpoint: https://router.huggingface.co
 Connected to Kubernetes cluster
 Watching for pods with schedulerName=ai-llama-scheduler

============================================================
 Detected pod: default/ai-test-pod-1
 Calling HuggingFace Llama-3.3-70B for scheduling decision...
 LLM decision: minikube-m03 (confidence: 0.85)
 Reasoning: Selected minikube-m03 due to lowest resource utilization...
 Bound pod default/ai-test-pod-1 to node minikube-m03
============================================================
```

### Pod Status:
```
NAME            STATUS    NODE
ai-test-pod-1   Running   minikube-m03
ai-test-pod-2   Running   minikube-m02
ai-test-pod-3   Running   minikube-m03
```

## Troubleshooting

### If you see "Using fallback":
- Check HuggingFace token is valid
- Verify internet connection
- Check HuggingFace API status: https://status.huggingface.co

### If pods stuck in "Pending":
- Verify scheduler is running
- Check scheduler logs for errors
- Ensure pods have `schedulerName: ai-llama-scheduler`

### If timeout errors:
- Already fixed! Just restart the scheduler with the updated code

## Quick Commands

```bash
# Verify setup
python verify_setup.py

# Start scheduler
python scheduler.py

# Create test pods
kubectl apply -f ai-test-pods.yaml

# Check pod status
kubectl get pods -o wide

# View scheduler stats (in scheduler logs)
# Look for the final statistics when you stop with Ctrl+C

# Clean up
kubectl delete -f ai-test-pods.yaml
```

## Next Steps After Testing

Once testing is successful:
1.  Commit to Git
2.  Push to GitHub
3.  Share with community
4.  Deploy to production cluster (optional)

---

**Ready to test!** 
