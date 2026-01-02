# Contributing to AI Kubernetes Scheduler

Thank you for your interest in contributing! 

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Include detailed steps to reproduce
- Provide scheduler logs and pod configurations

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Add type hints where appropriate
- Include docstrings for functions and classes
- Keep functions focused and concise

### Testing
- Test your changes with a local Kubernetes cluster
- Verify both LLM and fallback scheduling work
- Check edge cases (no nodes available, API failures, etc.)

### Documentation
- Update README.md if adding features
- Add inline comments for complex logic
- Update config.yaml.example if adding configuration options

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AI-Cloud-Scheduling.git
cd AI-Cloud-Scheduling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your HuggingFace token

# Start local Kubernetes cluster
minikube start --nodes 3

# Run scheduler
python scheduler.py
```

## Questions?

Feel free to open an issue for questions or discussions!
