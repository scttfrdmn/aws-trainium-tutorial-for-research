# Common Troubleshooting Scenarios

This guide covers the most frequently encountered issues when working with AWS Trainium and Inferentia, with step-by-step solutions and preventive measures.

## 🚀 Quick Start Troubleshooting

### Use the Interactive Tool
```bash
# Start interactive troubleshooter
python docs/troubleshooting/interactive_diagnosis.py

# Quick diagnosis for specific issues
python docs/troubleshooting/interactive_diagnosis.py --issue compilation

# Run system diagnostics
python docs/troubleshooting/interactive_diagnosis.py --diagnostics
```

## 📋 Top 10 Most Common Issues

### 1. "No Neuron devices found" Error

**Symptoms**:
```python
RuntimeError: No XLA devices found
```

**Diagnosis**:
- Wrong instance type (not trn1.* or inf2.*)
- Neuron driver not loaded
- Instance not fully initialized

**Solutions**:
```bash
# Check instance type
curl -s http://169.254.169.254/latest/meta-data/instance-type

# Verify Neuron devices
neuron-ls

# Restart Neuron service if needed
sudo systemctl restart neuron-discovery

# Check driver status
lsmod | grep neuron
```

**Prevention**:
- Always use trn1.* instances for training
- Always use inf2.* instances for inference
- Wait for instance to fully initialize before starting work

---

### 2. Model Compilation Timeout

**Symptoms**:
```
NeuronCompilationTimeoutError: Compilation timed out after 600 seconds
```

**Diagnosis**:
- Model too complex for default timeout
- Large model requiring more compilation time
- Network connectivity issues during compilation

**Solutions**:
```bash
# Increase compilation timeout
export NEURON_COMPILE_TIMEOUT=3600  # 1 hour

# Enable compilation caching
export NEURON_COMPILE_CACHE_URL=/tmp/neuron-cache

# Reduce batch size for compilation
# Compile with batch_size=1, then use larger batches at runtime
```

**Prevention**:
- Start with simple models to test setup
- Use incremental compilation approach
- Always enable compilation caching

---

### 3. Out of Memory (OOM) Errors

**Symptoms**:
```
RuntimeError: [E_MEMORY] Insufficient memory
```

**Diagnosis**:
- Batch size too large for available memory
- Model too large for instance type
- Memory leak in training loop

**Solutions**:
```bash
# Monitor memory usage
neuron-monitor

# Reduce batch size
batch_size = 1  # Start small and increase gradually

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache between iterations
torch.cuda.empty_cache()  # For GPU fallback
```

**Memory Optimization Techniques**:
```python
# Use mixed precision
with torch.autocast('cuda'):
    outputs = model(inputs)

# Gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Prevention**:
- Start with small batch sizes
- Monitor memory usage continuously
- Use appropriate instance sizes

---

### 4. torch-neuronx Import Errors

**Symptoms**:
```python
ModuleNotFoundError: No module named 'torch_neuronx'
```

**Diagnosis**:
- torch-neuronx not installed
- Incompatible PyTorch version
- Virtual environment issues

**Solutions**:
```bash
# Install torch-neuronx with correct index
pip install torch-neuronx==2.2.0 --extra-index-url https://pip.repos.neuron.amazonaws.com

# Check PyTorch compatibility
python -c "import torch; print(torch.__version__)"
# Should be PyTorch 2.9 for torch-neuronx 2.9.x (Neuron SDK 2.30, the last XLA-based version)

# Verify installation
python -c "import torch_neuronx; print('Success')"
```

**Prevention**:
- Always use the Neuron-specific PyTorch installation
- Check version compatibility before installing
- Use virtual environments to avoid conflicts

---

### 5. Slow Performance / Low Utilization

**Symptoms**:
- Model runs but much slower than expected
- Low Neuron core utilization
- CPU bottlenecks

**Diagnosis**:
- Model not compiled for Neuron
- Suboptimal batch size
- Data loading bottlenecks

**Solutions**:
```python
# Ensure proper Neuron compilation
import torch_neuronx
model_neuron = torch_neuronx.trace(model, example_inputs)

# Optimize batch size
# Try: 1, 4, 8, 16, 32 and measure throughput
batch_sizes = [1, 4, 8, 16, 32]
for bs in batch_sizes:
    # Benchmark and choose optimal size

# Use DataLoader with multiple workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# Enable performance profiling
export NEURON_PROFILE=/tmp/neuron_profile
```

**Performance Optimization**:
```python
# Use torch.jit.script for CPU preprocessing
@torch.jit.script
def preprocess(data):
    return data.float() / 255.0

# Async data loading
class AsyncDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        for batch in self.dataloader:
            with torch.cuda.stream(self.stream):
                yield batch.to(device, non_blocking=True)
```

---

### 6. Dynamic Shape Compilation Errors

**Symptoms**:
```
NeuronCompilationError: Dynamic shapes not supported
```

**Diagnosis**:
- Variable input sizes
- Dynamic control flow
- Runtime shape changes

**Solutions**:
```python
# Use fixed shapes for compilation
def get_fixed_shape_input():
    return torch.randn(1, 3, 224, 224)  # Fixed shape

# Pad sequences to fixed length
def pad_to_fixed_length(sequences, max_length=512):
    return torch.nn.functional.pad(sequences, (0, max_length - sequences.size(-1)))

# Compile multiple models for different shapes
models = {}
for batch_size in [1, 4, 8, 16]:
    example_input = torch.randn(batch_size, 3, 224, 224)
    models[batch_size] = torch_neuronx.trace(model, example_input)
```

**Prevention**:
- Design models with fixed input shapes
- Use padding for variable-length sequences
- Compile separate models for different shape ranges

---

### 7. Unsupported Operator Errors

**Symptoms**:
```
NeuronCompilationError: Unsupported operator: aten::some_operator
```

**Diagnosis**:
- Using operators not supported by Neuron
- Complex control flow
- Custom operators

**Solutions**:
```python
# Replace unsupported operators
# Instead of torch.unique, use alternative approaches
def replace_unique_with_workaround(tensor):
    # Use supported operations only
    pass

# Use torch.jit.script for complex operations
@torch.jit.script
def complex_operation(x):
    # This will run on CPU if not supported by Neuron
    return some_complex_computation(x)

# Partition model manually
def partition_model(model):
    # Split into Neuron-compatible and CPU parts
    neuron_part = NeuronCompatibleModel()
    cpu_part = CPUModel()
    return neuron_part, cpu_part
```

**Workarounds**:
```python
# Common operator replacements
def neuron_friendly_softmax(x, dim=-1):
    # Use explicit operations instead of F.softmax if needed
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def neuron_friendly_layernorm(x, eps=1e-5):
    # Manual layer norm implementation
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)
```

---

### 8. Unexpected High AWS Costs

**Symptoms**:
- AWS bill higher than expected
- Instances running longer than intended
- Data transfer charges

**Diagnosis**:
- Instances not terminated
- Forgetting to use spot instances
- Large data transfers

**Solutions**:
```bash
# Check running instances
aws ec2 describe-instances --filters 'Name=instance-state-name,Values=running'

# Terminate instances
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# Set up billing alerts
aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget file://budget.json

# Use spot instances
aws ec2 request-spot-instances --spot-price "0.50" --instance-count 1 \
    --type "one-time" --launch-specification file://launch-spec.json
```

**Cost Prevention Script**:
```python
import boto3
import time

def auto_terminate_after_inactivity(max_idle_minutes=30):
    """Auto-terminate instance after period of inactivity."""
    ec2 = boto3.client('ec2')
    instance_id = boto3.Session().region_name
    
    start_time = time.time()
    while True:
        # Check if work is being done (implement your logic)
        if is_idle():
            idle_time = time.time() - start_time
            if idle_time > max_idle_minutes * 60:
                ec2.terminate_instances(InstanceIds=[instance_id])
                break
        else:
            start_time = time.time()
        
        time.sleep(60)  # Check every minute
```

---

### 9. Data Loading Performance Issues

**Symptoms**:
- Training slow despite good model performance
- CPU utilization high during data loading
- GPU/Neuron idle time

**Diagnosis**:
- Inefficient data preprocessing
- Single-threaded data loading
- Large dataset on slow storage

**Solutions**:
```python
# Optimize DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # Use multiple processes
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)

# Use S3 optimization
import boto3
s3_client = boto3.client('s3', config=boto3.session.Config(
    max_pool_connections=50,  # Increase connection pool
    retries={'max_attempts': 3}
))

# Preprocess data offline
def preprocess_and_save(dataset, output_path):
    """Preprocess dataset once and save."""
    processed_data = []
    for item in dataset:
        processed_item = preprocess(item)
        processed_data.append(processed_item)
    
    torch.save(processed_data, output_path)
```

**S3 Data Loading Optimization**:
```python
import concurrent.futures
import boto3

class OptimizedS3Dataset:
    def __init__(self, bucket, prefix):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket
        self.keys = self._list_objects(prefix)
    
    def _list_objects(self, prefix):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        # Use concurrent downloads for batches
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return torch.load(obj['Body'])
```

---

### 10. Model Accuracy/Convergence Issues

**Symptoms**:
- Model not converging
- Different results compared to GPU
- Training instability

**Diagnosis**:
- Numerical precision differences
- Different operator implementations
- Learning rate needs adjustment

**Solutions**:
```python
# Use appropriate learning rates for Neuron
# Often need slightly different LR than GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Try lower LR

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Monitor for numerical issues
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Problematic gradient in {name}: {grad_norm}")

# Use mixed precision carefully
scaler = torch.cuda.amp.GradScaler()
```

**Debugging Convergence**:
```python
# Compare with CPU/GPU implementation
def compare_implementations():
    # Run same model on CPU and Neuron
    cpu_model = model.cpu()
    neuron_model = torch_neuronx.trace(model, example_input)
    
    cpu_output = cpu_model(test_input.cpu())
    neuron_output = neuron_model(test_input)
    
    diff = torch.abs(cpu_output - neuron_output.cpu())
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")
```

## 🔍 Diagnostic Commands

### System Health Check
```bash
# Check instance type
curl -s http://169.254.169.254/latest/meta-data/instance-type

# List Neuron devices
neuron-ls

# Monitor Neuron utilization
neuron-monitor

# Check Neuron driver
lsmod | grep neuron

# Verify torch-neuronx
python -c "import torch_neuronx; import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"
```

### Performance Monitoring
```bash
# Enable detailed profiling
export NEURON_PROFILE=/tmp/neuron_profile

# Monitor system resources
htop

# Check network usage
iftop

# Monitor disk I/O
iotop
```

### Debugging Environment
```bash
# Enable verbose compilation
export NEURON_COMPILE_VERBOSE=1

# Enable debug logging
export NEURON_RT_LOG_LEVEL=DEBUG

# Check environment variables
env | grep NEURON
```

## 🛠️ Advanced Troubleshooting

### Memory Debugging
```python
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Your code here
model_output = model(inputs)

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

### Compilation Debugging
```python
# Save compilation artifacts
import os
os.environ['NEURON_DUMP_HLO_SNAPSHOT'] = '/tmp/hlo_snapshots'

# Analyze compilation graph
def analyze_compilation(model, example_input):
    import torch_neuronx
    
    # Enable detailed compilation logs
    os.environ['NEURON_COMPILE_VERBOSE'] = '1'
    
    try:
        traced_model = torch_neuronx.trace(model, example_input)
        print("✅ Compilation successful")
        return traced_model
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        # Analyze the error and suggest fixes
        if "timeout" in str(e).lower():
            print("💡 Try increasing NEURON_COMPILE_TIMEOUT")
        elif "operator" in str(e).lower():
            print("💡 Check for unsupported operators")
        return None
```

## 📚 Additional Resources

### Documentation
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [AWS EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)

### Community Support
- [GitHub Issues](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues)
- [AWS Developer Forums](https://forums.aws.amazon.com/)
- [PyTorch Discussion Forums](https://discuss.pytorch.org/)

### Monitoring Tools
- [AWS CloudWatch](https://aws.amazon.com/cloudwatch/)
- [neuron-monitor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html)
- [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)

---

*For interactive troubleshooting, run: `python docs/troubleshooting/interactive_diagnosis.py`*