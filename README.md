# 🚀 AI-Powered Kubernetes Scheduler with Llama-3.3-70B

An intelligent Kubernetes scheduler that uses **Meta's Llama-3.3-70B-Instruct** model via HuggingFace Inference API to make smart pod placement decisions based on real-time cluster metrics.

## ✨ Features

- **🤖 LLM-Powered Scheduling**: Uses Llama-3.3-70B for intelligent node selection
- **📦 Request Caching**: Avoids redundant API calls for similar scheduling decisions
- **🔄 Retry Logic**: Exponential backoff for failed API requests
- **⚡ Circuit Breaker**: Prevents cascading failures during API outages
- **📊 Metrics & Monitoring**: Comprehensive statistics and performance tracking
- **🎯 Smart Fallback**: Multiple fallback strategies when LLM is unavailable
- **⚙️ Configurable**: YAML-based configuration for easy customization

## 🏗️ Architecture

```
┌─────────────────┐
│  Unscheduled    │
│     Pods        │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────────┐
│   AI Scheduler  │─────▶│  HuggingFace API │
│  (scheduler.py) │      │  Llama-3.3-70B   │
└────────┬────────┘      └──────────────────┘
         │
         v
┌─────────────────┐
│  Kubernetes     │
│  Nodes          │
└─────────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Kubernetes cluster** (Minikube, Kind, or production cluster)
- **HuggingFace API token** ([Get one here](https://huggingface.co/settings/tokens))
- **kubectl** configured to access your cluster

## 🚀 Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Cloud-Scheduling

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your HuggingFace token
# HUGGINGFACE_TOKEN=your_token_here
```

### 3. Start Kubernetes Cluster

```bash
# Using Minikube (example with 3 nodes)
minikube start --nodes 3 --driver=docker

# Verify nodes are ready
kubectl get nodes
```

### 4. Run the Scheduler

```bash
# Start the AI scheduler
python scheduler.py
```

You should see:
```
🚀 AI-Powered Kubernetes Scheduler with Llama-3.3-70B
============================================================
Using HuggingFace Inference API
Model: meta-llama/Llama-3.3-70B-Instruct
============================================================
✅ Initialized HuggingFace client for model: meta-llama/Llama-3.3-70B-Instruct
📡 Endpoint: https://router.huggingface.co
✅ Connected to Kubernetes cluster
👀 Watching for pods with schedulerName=ai-llama-scheduler
```

### 5. Test with Sample Pods

```bash
# In a new terminal, create test pods
kubectl apply -f ai-test-pods.yaml

# Watch the scheduler make decisions
# Check pod status
kubectl get pods -o wide
```

## ⚙️ Configuration

Edit `config.yaml` to customize scheduler behavior:

```yaml
scheduler:
  name: ai-llama-scheduler
  
llm:
  model: meta-llama/Llama-3.3-70B-Instruct
  temperature: 0.3
  max_tokens: 200
  max_retries: 3
  
cache:
  enabled: true
  ttl: 300  # 5 minutes
  
circuit_breaker:
  enabled: true
  failure_threshold: 5
  timeout: 60
```

## 📁 Project Structure

```
AI-Cloud-Scheduling/
├── scheduler.py           # Main scheduler implementation
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
├── ai-test-pods.yaml    # Sample test pods
└── README.md            # This file
```

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_TOKEN` | HuggingFace API token (required) | - |
| `SCHEDULER_NAME` | Name of the scheduler | `ai-llama-scheduler` |
| `LLM_MODEL` | Model to use | `meta-llama/Llama-3.3-70B-Instruct` |
| `LLM_ENDPOINT` | HuggingFace endpoint | `https://router.huggingface.co` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format (text/json) | `text` |

## 📊 How It Works

1. **Pod Detection**: Scheduler watches for pods with `schedulerName: ai-llama-scheduler`
2. **Context Collection**: Gathers real-time metrics from all cluster nodes
3. **Prompt Construction**: Creates a structured prompt with pod requirements and node states
4. **LLM Decision**: Calls Llama-3.3-70B via HuggingFace API for intelligent node selection
5. **Validation**: Ensures selected node is valid and available
6. **Binding**: Binds the pod to the selected node
7. **Caching**: Caches decision for similar future requests

## 🎯 Fallback Strategies

When the LLM is unavailable, the scheduler uses configurable fallback strategies:

- **`resource_balanced`** (default): Balances CPU, memory, and pod count
- **`least_loaded`**: Selects node with lowest resource usage
- **`round_robin`**: Distributes pods evenly across nodes

## 📈 Monitoring

The scheduler tracks comprehensive metrics:

- Total scheduling requests
- LLM vs fallback decisions
- Cache hit rate
- Average response time
- Circuit breaker trips
- Failed bindings

Access stats in the logs or extend with Prometheus metrics.

## 🐛 Troubleshooting

### "HUGGINGFACE_TOKEN environment variable is required"
- Get a token from https://huggingface.co/settings/tokens
- Add it to your `.env` file or export it: `export HUGGINGFACE_TOKEN=your_token`

### "Cannot connect to Kubernetes cluster"
- Verify kubectl is configured: `kubectl cluster-info`
- Check kubeconfig: `echo $KUBECONFIG`

### "Circuit breaker is OPEN"
- HuggingFace API is experiencing issues
- Scheduler will use fallback strategy automatically
- Check HuggingFace status: https://status.huggingface.co

### Pods stuck in "Pending"
- Verify scheduler is running
- Check pod has correct `schedulerName: ai-llama-scheduler`
- View scheduler logs for errors

## 🔒 Security Best Practices

- **Never commit `.env` file** to version control
- Store HuggingFace token securely (use secrets management in production)
- Use RBAC to limit scheduler permissions
- Monitor API usage to avoid rate limits

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Additional Resources

- [HuggingFace Inference API Docs](https://huggingface.co/docs/api-inference/)
- [Kubernetes Scheduler Documentation](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/)
- [Llama-3.3-70B Model Card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

## 🙏 Acknowledgments

- Meta AI for Llama-3.3-70B model
- HuggingFace for Inference API
- Kubernetes community

---

**Made with ❤️ using AI-powered scheduling**
