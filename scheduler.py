"""
AI-Powered Kubernetes Scheduler with Llama-3.3-70B
Uses HuggingFace Inference API for intelligent pod scheduling decisions
"""

import sys
import os
import json
import time
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
log_format = os.getenv('LOG_FORMAT', 'text')

if log_format == 'json':
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(message)s'
    )
else:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)


# Configuration
# ============================================================================
def load_config() -> Dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()
SCHEDULER_NAME = os.getenv('SCHEDULER_NAME', CONFIG.get('scheduler', {}).get('name', 'ai-llama-scheduler'))
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
LLM_MODEL = os.getenv('LLM_MODEL', CONFIG.get('llm', {}).get('model', 'meta-llama/Llama-3.3-70B-Instruct'))
LLM_ENDPOINT = os.getenv('LLM_ENDPOINT', CONFIG.get('llm', {}).get('endpoint', 'https://router.huggingface.co'))
REQUEST_TIMEOUT = int(os.getenv('LLM_TIMEOUT', CONFIG.get('llm', {}).get('timeout', 60)))
MAX_RETRIES = int(os.getenv('LLM_MAX_RETRIES', CONFIG.get('llm', {}).get('max_retries', 3)))

if not HUGGINGFACE_TOKEN:
    logger.error("HUGGINGFACE_TOKEN environment variable is required!")
    logger.error("Get your token from: https://huggingface.co/settings/tokens")
    logger.error("Set it with: export HUGGINGFACE_TOKEN=your_token_here")
    sys.exit(1)


# ============================================================================
# Data Models
# ============================================================================
@dataclass
class NodeMetrics:
    """Represents current metrics for a Kubernetes node"""
    name: str
    cpu_usage_percent: float
    memory_usage_percent: float
    available_cpu: float
    available_memory: float
    pod_count: int
    max_pods: int
    labels: Dict[str, str]
    taints: List[Dict[str, str]]
    conditions: List[Dict[str, str]]

@dataclass
class PodSpec:
    """Represents a pod scheduling request"""
    name: str
    namespace: str
    cpu_request: float
    memory_request: float
    node_selector: Dict[str, str]
    tolerations: List[Dict[str, str]]
    affinity_rules: Dict[str, Any]
    priority: int

@dataclass
class SchedulingDecision:
    """Represents the scheduling decision"""
    selected_node: str
    confidence: float
    reasoning: str
    fallback_needed: bool = False

# ============================================================================
# Context Manager - Collects Cluster State
# ============================================================================
class ContextManager:
    """Collects and manages live cluster metrics"""
    
    def __init__(self):
        try:
            config.load_kube_config()
            self.v1 = client.CoreV1Api()
            logger.info(" Connected to Kubernetes cluster")
        except Exception as e:
            logger.error(f"Failed to load kubeconfig: {e}")
            raise
    
    def get_node_metrics(self) -> List[NodeMetrics]:
        """Collect current metrics for all nodes"""
        try:
            nodes = self.v1.list_node()
            node_metrics = []
            
            for node in nodes.items:
                node_name = node.metadata.name
                labels = node.metadata.labels or {}
                taints = [{"key": t.key, "effect": t.effect, "value": t.value or ""}
                         for t in (node.spec.taints or [])]
                conditions = [{"type": c.type, "status": c.status, "reason": c.reason or ""}
                            for c in (node.status.conditions or [])]
                
                # Get resource capacity
                capacity = node.status.capacity
                allocatable = node.status.allocatable
                
                available_cpu = self._parse_cpu(allocatable.get('cpu', '0'))
                available_memory = self._parse_memory(allocatable.get('memory', '0'))
                max_pods = int(allocatable.get('pods', '0'))
                
                # Get current pod count
                pod_list = self.v1.list_pod_for_all_namespaces(
                    field_selector=f"spec.nodeName={node_name}"
                )
                pod_count = len(pod_list.items)
                
                # Calculate usage (simplified - in production use metrics-server)
                cpu_usage = (pod_count / max_pods) * 50 if max_pods > 0 else 0
                memory_usage = (pod_count / max_pods) * 50 if max_pods > 0 else 0
                
                node_metrics.append(NodeMetrics(
                    name=node_name,
                    cpu_usage_percent=cpu_usage,
                    memory_usage_percent=memory_usage,
                    available_cpu=available_cpu,
                    available_memory=available_memory,
                    pod_count=pod_count,
                    max_pods=max_pods,
                    labels=labels,
                    taints=taints,
                    conditions=conditions
                ))
            
            return node_metrics
            
        except Exception as e:
            logger.error(f"Error collecting node metrics: {e}")
            return []
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to cores"""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to GB"""
        if memory_str.endswith('Ki'):
            return float(memory_str[:-2]) / 1024 / 1024
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2])
        else:
            return float(memory_str) / 1024 / 1024 / 1024

# ============================================================================
# Prompt Engine - Constructs Prompts
# ============================================================================
class PromptEngine:
    """Constructs structured prompts for the LLM"""
    
    def __init__(self):
        self.system_prompt = """You are an intelligent Kubernetes scheduler AI. Your task is to select the BEST ACTUAL node from the available nodes list for pod placement.

CRITICAL RULES:
1. You MUST select a node name from the "AVAILABLE NODES" list below
2. The selected_node value MUST be EXACTLY one of the node names
3. Return ONLY valid JSON, nothing else

Response format:
{
    "selected_node": "node-name",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}

Selection criteria:
- Lowest resource utilization (CPU + Memory)
- Available pod capacity
- Resource requests fit available resources
- Node health and readiness"""
    
    def construct_scheduling_prompt(self, pod_spec: PodSpec, node_metrics: List[NodeMetrics]) -> str:
        """Create a structured prompt for scheduling decision"""
        
        pod_info = f"""
POD TO SCHEDULE:
Name: {pod_spec.name}
Namespace: {pod_spec.namespace}
CPU Request: {pod_spec.cpu_request} cores
Memory Request: {pod_spec.memory_request} GB
Priority: {pod_spec.priority}
"""
        
        cluster_info = "AVAILABLE NODES:\n"
        node_names = []
        for node in node_metrics:
            node_names.append(node.name)
            available_cpu = node.available_cpu * (100 - node.cpu_usage_percent) / 100.0
            available_memory = node.available_memory * (100 - node.memory_usage_percent) / 100.0
            
            cluster_info += f"""
{node.name}:
  - CPU Usage: {node.cpu_usage_percent:.1f}% (available: {available_cpu:.2f} cores)
  - Memory Usage: {node.memory_usage_percent:.1f}% (available: {available_memory:.2f} GB)
  - Pods: {node.pod_count}/{node.max_pods}
  - Status: Ready
"""
        
        cluster_info += f"\nVALID NODE NAMES: {', '.join(node_names)}\n"
        
        full_prompt = f"""{self.system_prompt}

{pod_info}
{cluster_info}

Select the best node from [{', '.join(node_names)}] and respond with JSON only:"""
        
        return full_prompt

# ============================================================================
# Request Cache - Avoid Redundant LLM Calls
# ============================================================================
class RequestCache:
    """Simple in-memory cache for scheduling decisions"""
    
    def __init__(self, ttl: int = 300, max_size: int = 100):
        self.cache: Dict[str, Tuple[SchedulingDecision, datetime]] = {}
        self.ttl = ttl
        self.max_size = max_size
    
    def _generate_key(self, pod_spec: 'PodSpec', node_metrics: List['NodeMetrics']) -> str:
        """Generate cache key from pod spec and node states"""
        pod_data = f"{pod_spec.cpu_request}_{pod_spec.memory_request}_{pod_spec.priority}"
        nodes_data = "_".join([f"{n.name}_{n.cpu_usage_percent:.1f}_{n.memory_usage_percent:.1f}" 
                               for n in sorted(node_metrics, key=lambda x: x.name)])
        key = hashlib.md5(f"{pod_data}_{nodes_data}".encode()).hexdigest()
        return key
    
    def get(self, pod_spec: 'PodSpec', node_metrics: List['NodeMetrics']) -> Optional[SchedulingDecision]:
        """Get cached decision if available and not expired"""
        key = self._generate_key(pod_spec, node_metrics)
        if key in self.cache:
            decision, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                logger.debug(f"Cache hit for key {key}")
                return decision
            else:
                del self.cache[key]
        return None
    
    def set(self, pod_spec: 'PodSpec', node_metrics: List['NodeMetrics'], decision: SchedulingDecision):
        """Cache a scheduling decision"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self._generate_key(pod_spec, node_metrics)
        self.cache[key] = (decision, datetime.now())
        logger.debug(f"Cached decision for key {key}")

# ============================================================================
# Circuit Breaker - Prevent Cascading Failures
# ============================================================================
class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                logger.info("Circuit breaker entering HALF_OPEN state")
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                logger.info("Circuit breaker closing")
                self.state = 'CLOSED'
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now()
            
            if self.failures >= self.failure_threshold:
                logger.error(f"Circuit breaker opening after {self.failures} failures")
                self.state = 'OPEN'
            raise e

# ============================================================================
# HuggingFace Client - Calls HuggingFace Inference API
# ============================================================================
class HuggingFaceClient:
    """Client to communicate with HuggingFace Inference API for Llama-3.3-70B"""
    
    def __init__(self, model: str, token: str, endpoint: str):
        self.model = model
        self.endpoint = endpoint
        self.client = InferenceClient(token=token, base_url=endpoint)
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_requests": 0,
            "avg_response_time": 0.0,
            "circuit_breaker_trips": 0
        }
        
        # Initialize cache and circuit breaker
        cache_config = CONFIG.get('cache', {})
        self.cache = RequestCache(
            ttl=cache_config.get('ttl', 300),
            max_size=cache_config.get('max_size', 100)
        ) if cache_config.get('enabled', True) else None
        
        cb_config = CONFIG.get('circuit_breaker', {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get('failure_threshold', 5),
            timeout=cb_config.get('timeout', 60)
        ) if cb_config.get('enabled', True) else None
        
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to HuggingFace service"""
        try:
            logger.info(f" Initialized HuggingFace client for model: {self.model}")
            logger.info(f" Endpoint: {self.endpoint}")
        except Exception as e:
            logger.error(f" Cannot initialize HuggingFace client: {e}")
            raise
    
    def get_scheduling_decision(self, prompt: str, pod_spec: 'PodSpec', node_metrics: List['NodeMetrics']) -> SchedulingDecision:
        """Call HuggingFace API for scheduling decision with caching and retry logic"""
        # Check cache first
        if self.cache:
            cached_decision = self.cache.get(pod_spec, node_metrics)
            if cached_decision:
                self.stats["cached_requests"] += 1
                logger.info(" Using cached decision")
                return cached_decision
        
        self.stats["total_requests"] += 1
        
        # Try with circuit breaker and retries
        for attempt in range(MAX_RETRIES):
            try:
                if self.circuit_breaker:
                    decision = self.circuit_breaker.call(self._make_api_call, prompt, node_metrics)
                else:
                    decision = self._make_api_call(prompt, node_metrics)
                
                # Cache successful decision
                if self.cache and not decision.fallback_needed:
                    self.cache.set(pod_spec, node_metrics, decision)
                
                return decision
                
            except Exception as e:
                if "Circuit breaker is OPEN" in str(e):
                    self.stats["circuit_breaker_trips"] += 1
                    logger.warning(" Circuit breaker is OPEN, using fallback")
                    return self._fallback_decision(node_metrics, "Circuit breaker open")
                
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {MAX_RETRIES} attempts failed: {e}")
                    self.stats["failed_requests"] += 1
                    return self._fallback_decision(node_metrics, f"All retries failed: {e}")
    
    def _make_api_call(self, prompt: str, node_metrics: List['NodeMetrics']) -> SchedulingDecision:
        """Make the actual API call to HuggingFace"""
        start_time = time.time()
        
        try:
            # Call HuggingFace Inference API
            llm_config = CONFIG.get('llm', {})
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an intelligent Kubernetes scheduler. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=llm_config.get('max_tokens', 200),
                temperature=llm_config.get('temperature', 0.3)
            )
            
            response_time = time.time() - start_time
            
            # Update average response time
            total = self.stats["total_requests"]
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (total - 1) + response_time) / total
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            logger.debug(f"LLM Response: {response_text}")
            
            # Parse JSON from response
            decision_data = self._extract_json(response_text)
            
            if decision_data:
                selected_node = decision_data.get("selected_node", "")
                
                # Validate node exists
                available_nodes = [n.name for n in node_metrics]
                if selected_node in available_nodes:
                    self.stats["successful_requests"] += 1
                    return SchedulingDecision(
                        selected_node=selected_node,
                        confidence=decision_data.get("confidence", 0.8),
                        reasoning=decision_data.get("reasoning", "LLM decision"),
                        fallback_needed=False
                    )
                else:
                    logger.warning(f"LLM selected invalid node: {selected_node}")
                    return self._fallback_decision(node_metrics, "Invalid node selected")
            else:
                logger.error("Could not parse JSON from LLM response")
                return self._fallback_decision(node_metrics, "JSON parsing failed")
                
        except Exception as e:
            logger.error(f"Error calling HuggingFace API: {e}")
            raise
    
    def _extract_json(self, response: str) -> Optional[Dict]:
        """Extract JSON from response with multiple strategies"""
        # Strategy 1: Find JSON code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 2: Find last complete JSON object
        json_start = response.rfind('{')
        if json_start != -1:
            brace_count = 0
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[json_start:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
        
        # Strategy 3: Find first JSON object
        json_start = response.find('{')
        if json_start != -1:
            brace_count = 0
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[json_start:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
        
        return None
    
    def _fallback_decision(self, node_metrics: List['NodeMetrics'], reason: str) -> SchedulingDecision:
        """Fallback scheduling using simple heuristic"""
        if not node_metrics:
            return SchedulingDecision("", 0.0, "No nodes available", True)
        
        fallback_strategy = CONFIG.get('fallback', {}).get('strategy', 'resource_balanced')
        
        best_node = None
        best_score = -1
        
        for node in node_metrics:
            is_ready = any(c.get('type') == 'Ready' and c.get('status') == 'True'
                          for c in node.conditions)
            if not is_ready:
                continue
            
            if fallback_strategy == 'resource_balanced':
                cpu_score = (100 - node.cpu_usage_percent) / 100.0
                memory_score = (100 - node.memory_usage_percent) / 100.0
                pod_score = (node.max_pods - node.pod_count) / node.max_pods if node.max_pods > 0 else 0
                score = (cpu_score * 0.35) + (memory_score * 0.35) + (pod_score * 0.30)
            elif fallback_strategy == 'least_loaded':
                score = (100 - node.cpu_usage_percent) + (100 - node.memory_usage_percent)
            else:  # round_robin or default
                score = node.pod_count  # Prefer nodes with fewer pods
            
            if score > best_score:
                best_score = score
                best_node = node
        
        if best_node:
            return SchedulingDecision(
                selected_node=best_node.name,
                confidence=0.4,
                reasoning=f"Fallback ({fallback_strategy}): {reason}",
                fallback_needed=True
            )
        else:
            return SchedulingDecision("", 0.0, f"Fallback failed: {reason}", True)
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return self.stats.copy()

# ============================================================================
# Integration Layer - Binds Pods to Nodes
# ============================================================================
class IntegrationLayer:
    """Handles Kubernetes API integration"""
    
    def __init__(self):
        try:
            config.load_kube_config()
            self.v1 = client.CoreV1Api()
        except Exception as e:
            logger.error(f"Failed to load kubeconfig: {e}")
            raise
    
    def bind_pod_to_node(self, pod_name: str, namespace: str, node_name: str) -> bool:
        """Bind pod to selected node"""
        try:
            # Create proper Binding object
            binding = client.V1Binding(
                api_version="v1",
                kind="Binding",
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace
                ),
                target=client.V1ObjectReference(
                    api_version="v1",
                    kind="Node",
                    name=node_name
                )
            )
            
            # Bind the pod - FIXED: removed 'name' parameter
            self.v1.create_namespaced_binding(
                namespace=namespace,
                body=binding,
                _preload_content=False
            )
            
            logger.info(f" Bound pod {namespace}/{pod_name} to node {node_name}")
            return True
            
        except ApiException as e:
            logger.error(f" Error binding pod {pod_name} to node {node_name}: {e}")
            if e.body:
                try:
                    error_body = json.loads(e.body)
                    logger.error(f"API Error: {error_body.get('message', 'No message')}")
                except json.JSONDecodeError:
                    logger.error(f"API Error Body: {e.body}")
            return False
        except Exception as e:
            logger.error(f" Unexpected error binding pod: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================================================
# Custom Scheduler - Main Orchestration
# ============================================================================
class CustomScheduler:
    """Main custom scheduler service"""
    
    def __init__(self, scheduler_name: str):
        self.scheduler_name = scheduler_name
        self.context_manager = ContextManager()
        self.prompt_engine = PromptEngine()
        self.llm_client = HuggingFaceClient(LLM_MODEL, HUGGINGFACE_TOKEN, LLM_ENDPOINT)
        self.integration_layer = IntegrationLayer()
        self.running = False
        self.stats = {
            "total_scheduled": 0,
            "llm_decisions": 0,
            "fallback_decisions": 0,
            "failed_bindings": 0
        }
    
    async def start(self):
        """Start the scheduler service"""
        self.running = True
        logger.info(f" Starting {self.scheduler_name}...")
        logger.info(f" Watching for pods with schedulerName={self.scheduler_name}")
        await self._watch_pods()
    
    def stop(self):
        """Stop the scheduler service"""
        self.running = False
        logger.info("⏹  Scheduler stopped")
    
    async def _watch_pods(self):
        """Watch for pods that need scheduling"""
        try:
            v1 = client.CoreV1Api()
            w = watch.Watch()
            
            logger.info(f" Watching for unscheduled pods...")
            
            while self.running:
                try:
                    for event in w.stream(
                        v1.list_pod_for_all_namespaces,
                        timeout_seconds=60
                    ):
                        if not self.running:
                            break
                        
                        pod = event['object']
                        
                        # Check if this pod needs our scheduler
                        if (pod.status.phase == 'Pending' and
                            pod.spec.scheduler_name == self.scheduler_name and
                            pod.spec.node_name is None):
                            
                            logger.info(f"\n{'='*60}")
                            logger.info(f" Detected pod: {pod.metadata.namespace}/{pod.metadata.name}")
                            await self._schedule_pod(pod)
                            logger.info(f"{'='*60}\n")
                
                except Exception as e:
                    logger.error(f"Watch error: {e}")
                    await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"Fatal watch error: {e}")
    
    async def _schedule_pod(self, pod):
        """Schedule a Kubernetes pod"""
        pod_spec = self._convert_pod_to_spec(pod)
        node_metrics = self.context_manager.get_node_metrics()
        
        if not node_metrics:
            logger.error(" No available nodes")
            return
        
        # Construct prompt
        prompt = self.prompt_engine.construct_scheduling_prompt(pod_spec, node_metrics)
        
        # Get decision from HuggingFace LLM
        logger.info(" Calling HuggingFace Llama-3.3-70B for scheduling decision...")
        decision = self.llm_client.get_scheduling_decision(prompt, pod_spec, node_metrics)
        
        if decision.fallback_needed:
            self.stats["fallback_decisions"] += 1
            logger.warning(f"  Using fallback: {decision.reasoning}")
        else:
            self.stats["llm_decisions"] += 1
            logger.info(f" LLM decision: {decision.selected_node} (confidence: {decision.confidence:.2f})")
        
        logger.info(f" Reasoning: {decision.reasoning}")
        
        # Bind pod to node
        if decision.selected_node:
            success = self.integration_layer.bind_pod_to_node(
                pod.metadata.name,
                pod.metadata.namespace,
                decision.selected_node
            )
            
            if success:
                self.stats["total_scheduled"] += 1
            else:
                self.stats["failed_bindings"] += 1
        else:
            logger.error(" No node selected")
            self.stats["failed_bindings"] += 1
    
    def _convert_pod_to_spec(self, pod) -> PodSpec:
        """Convert Kubernetes pod object to PodSpec"""
        containers = pod.spec.containers or []
        cpu_request = 0.0
        memory_request = 0.0
        
        for container in containers:
            if container.resources and container.resources.requests:
                cpu_str = container.resources.requests.get('cpu', '0')
                memory_str = container.resources.requests.get('memory', '0')
                
                if isinstance(cpu_str, str) and cpu_str.endswith('m'):
                    cpu_request += float(cpu_str[:-1]) / 1000
                else:
                    cpu_request += float(cpu_str or 0)
                
                if isinstance(memory_str, str):
                    if memory_str.endswith('Ki'):
                        memory_request += float(memory_str[:-2]) / 1024 / 1024
                    elif memory_str.endswith('Mi'):
                        memory_request += float(memory_str[:-2]) / 1024
                    elif memory_str.endswith('Gi'):
                        memory_request += float(memory_str[:-2])
        
        return PodSpec(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            cpu_request=cpu_request,
            memory_request=memory_request,
            node_selector=pod.spec.node_selector or {},
            tolerations=pod.spec.tolerations or [],
            affinity_rules={},
            priority=pod.spec.priority or 0
        )
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        stats = self.stats.copy()
        stats.update({"llm_client": self.llm_client.get_stats()})
        return stats

# ============================================================================
# Main Function
# ============================================================================
async def main():
    """Main function to run the AI scheduler"""
    print(" AI-Powered Kubernetes Scheduler with Llama-3.3-70B")
    print("=" * 60)
    print("Using HuggingFace Inference API")
    print(f"Model: {LLM_MODEL}")
    print("=" * 60)
    
    # Create scheduler
    scheduler = CustomScheduler(SCHEDULER_NAME)
    
    try:
        print("\n Scheduler initialized successfully!")
        print(f" Watching for pods with schedulerName={SCHEDULER_NAME}")
        print("\nPress Ctrl+C to stop...\n")
        
        # Start scheduler
        await scheduler.start()
        
    except KeyboardInterrupt:
        print("\n\n⏹  Scheduler stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scheduler.stop()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print(" Final Statistics:")
        print("=" * 60)
        stats = scheduler.get_stats()
        print(f"Total Scheduled: {stats['total_scheduled']}")
        print(f"LLM Decisions: {stats['llm_decisions']}")
        print(f"Fallback Decisions: {stats['fallback_decisions']}")
        print(f"Failed Bindings: {stats['failed_bindings']}")
        
        llm_stats = stats.get('llm_client', {})
        print(f"\nLLM Client Stats:")
        print(f"  Total Requests: {llm_stats.get('total_requests', 0)}")
        print(f"  Successful: {llm_stats.get('successful_requests', 0)}")
        print(f"  Failed: {llm_stats.get('failed_requests', 0)}")
        print(f"  Avg Response Time: {llm_stats.get('avg_response_time', 0):.2f}s")
        print("=" * 60)

if __name__ == "__main__":
    print("Starting scheduler...\n")
    asyncio.run(main())