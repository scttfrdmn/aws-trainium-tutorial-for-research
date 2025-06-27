# Comprehensive Error Handling and Debugging Guide

This guide provides systematic approaches to diagnosing, debugging, and resolving issues when working with AWS Trainium and Inferentia. It covers common problems, advanced debugging techniques, and production troubleshooting workflows.

**Last Updated**: 2024-12-19
**Tested Versions**: Neuron SDK 2.19.0, torch-neuronx 2.1.0, PyTorch 2.3.1

## 🚨 Quick Emergency Troubleshooting

### Critical Issues (System Down)
```bash
# 1. Emergency health check
python -c "import torch_neuronx; print('✅ Neuron accessible')"

# 2. Instance status check
python -c "import torch_xla.core.xla_model as xm; print('Devices:', xm.get_xla_supported_devices())"

# 3. Memory emergency cleanup
python -c "import torch_xla.core.xla_model as xm; xm.wait_device_ops(); print('✅ Memory cleared')"

# 4. Emergency instance restart (last resort)
sudo systemctl restart neuron-monitor
sudo systemctl restart neuron-discovery
```

### Quick Health Diagnostics
```bash
# System health overview
neuron-ls  # List Neuron devices
neuron-top  # Real-time monitoring
neuron-monitor  # Comprehensive status

# Process check
ps aux | grep neuron  # Check Neuron processes
nvidia-ml-py3 || echo "Neuron-only system"  # Verify no GPU conflicts
```

## 🔍 Error Categories and Solutions

### 1. Installation and Environment Issues

#### **Error**: `ModuleNotFoundError: No module named 'torch_neuronx'`
```bash
# Diagnosis
pip list | grep neuron
python --version

# Solutions (in order of preference)
# Option 1: Clean reinstall
pip uninstall torch-neuronx torch-xla
pip install torch-neuronx==2.1.0 --index-url https://pip.repos.neuron.amazonaws.com

# Option 2: Virtual environment reset
python -m venv neuron-env-new
source neuron-env-new/bin/activate
pip install -r requirements-exact.txt

# Option 3: System-wide installation
sudo pip install torch-neuronx==2.1.0 --index-url https://pip.repos.neuron.amazonaws.com
```

#### **Error**: `ImportError: /lib64/libc.so.6: version GLIBC_2.XX not found`
```bash
# Check glibc version
ldd --version

# Solutions
# Amazon Linux 2
sudo yum update glibc

# Ubuntu/Debian
sudo apt update && sudo apt upgrade libc6

# RHEL/CentOS
sudo yum update glibc

# If still failing, use compatible AMI
# Use AWS Deep Learning AMI with pre-installed Neuron
```

#### **Error**: `RuntimeError: Cannot initialize CUDA or NEURON device`
```bash
# Check instance type
curl -s http://169.254.169.254/latest/meta-data/instance-type

# Verify Neuron support
if [[ $(curl -s http://169.254.169.254/latest/meta-data/instance-type) =~ ^(trn1|inf2) ]]; then
    echo "✅ Neuron-supported instance"
else
    echo "❌ Use trn1.* or inf2.* instance types"
fi

# Check Neuron runtime
neuron-ls -v
```

### 2. Model Compilation Issues

#### **Error**: `NeuronCompilerError: Model compilation failed`
```python
# Enhanced compilation with debugging
import torch_neuronx
import logging

# Enable verbose compilation logging
logging.basicConfig(level=logging.DEBUG)

try:
    compiled_model = torch_neuronx.trace(
        model,
        example_input,
        compiler_args=[
            "--verbose=35",  # Maximum verbosity
            "--model-type=transformer",
            "--enable-saturate-infinity",
            "--neuroncore-pipeline-cores=1",  # Start with 1 core
            "--optimization-level=1"  # Conservative optimization
        ],
        compiler_timeout=300  # 5-minute timeout
    )
except Exception as e:
    print(f"Compilation error: {e}")
    # Fallback to CPU
    model_cpu = model.cpu()
    print("Running on CPU fallback")
```

#### **Error**: `Memory allocation failed during compilation`
```python
# Memory-efficient compilation strategies
def compile_with_memory_management(model, example_input):
    """Compile model with progressive memory strategies."""

    strategies = [
        # Strategy 1: Reduce batch size
        {
            "batch_size": 1,
            "compiler_args": ["--model-type=transformer", "--optimization-level=1"]
        },
        # Strategy 2: Gradient checkpointing
        {
            "batch_size": 2,
            "compiler_args": ["--enable-gradient-checkpointing", "--optimization-level=1"]
        },
        # Strategy 3: Memory pooling
        {
            "batch_size": 4,
            "compiler_args": ["--enable-memory-pooling", "--optimization-level=2"]
        }
    ]

    for i, strategy in enumerate(strategies):
        try:
            print(f"Trying compilation strategy {i+1}/3...")

            # Adjust input batch size
            adjusted_input = example_input[:strategy["batch_size"]]

            compiled_model = torch_neuronx.trace(
                model,
                adjusted_input,
                compiler_args=strategy["compiler_args"]
            )

            print(f"✅ Compilation successful with strategy {i+1}")
            return compiled_model, strategy["batch_size"]

        except RuntimeError as e:
            print(f"Strategy {i+1} failed: {e}")
            continue

    raise RuntimeError("All compilation strategies failed")

# Usage
try:
    compiled_model, batch_size = compile_with_memory_management(model, example_input)
    print(f"Use batch_size={batch_size} for inference")
except Exception as e:
    print(f"Compilation completely failed: {e}")
```

#### **Error**: `Unsupported operation in Neuron compiler`
```python
# Operation compatibility checker
def check_model_compatibility(model, example_input):
    """Check model operations for Neuron compatibility."""

    unsupported_ops = []
    supported_alternatives = {
        "torch.nn.functional.interpolate": "Use fixed-size operations",
        "torch.nonzero": "Use masked operations",
        "dynamic shapes": "Use padding to fixed shapes",
        "torch.unique": "Implement with sorting",
        "advanced indexing": "Use gather/scatter operations"
    }

    # Trace model operations
    with torch.no_grad():
        try:
            # Dry run to identify operations
            _ = model(example_input)
        except Exception as e:
            print(f"Model execution error: {e}")

    # Check for known incompatible patterns
    model_str = str(model)

    for op, alternative in supported_alternatives.items():
        if op in model_str.lower():
            print(f"⚠️ Found potentially unsupported operation: {op}")
            print(f"   Suggested alternative: {alternative}")

    return unsupported_ops

# Model adaptation for Neuron
def adapt_model_for_neuron(model):
    """Adapt model to be Neuron-compatible."""

    # Replace unsupported operations
    for name, module in model.named_modules():
        # Replace interpolation with fixed-size alternatives
        if isinstance(module, torch.nn.Upsample):
            print(f"Replacing Upsample in {name}")
            # Implement fixed-size upsampling

        # Replace dynamic operations
        if hasattr(module, 'dynamic_shapes') and module.dynamic_shapes:
            print(f"Disabling dynamic shapes in {name}")
            module.dynamic_shapes = False

    return model
```

### 3. Training Issues

#### **Error**: `Loss becomes NaN during training`
```python
# Comprehensive NaN debugging and prevention
class NaNDebugger:
    """Debug and prevent NaN values during training."""

    def __init__(self, model, check_frequency=10):
        self.model = model
        self.check_frequency = check_frequency
        self.step_count = 0

    def check_for_nans(self, loss, inputs, outputs):
        """Check for NaN values in model components."""
        issues = []

        # Check loss
        if torch.isnan(loss).any():
            issues.append("Loss contains NaN")

        # Check inputs
        if torch.isnan(inputs).any():
            issues.append("Input contains NaN")

        # Check outputs
        if torch.isnan(outputs).any():
            issues.append("Output contains NaN")

        # Check model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"Parameter {name} contains NaN")
            if param.grad is not None and torch.isnan(param.grad).any():
                issues.append(f"Gradient {name} contains NaN")

        return issues

    def debug_step(self, loss, inputs, outputs, optimizer):
        """Debug a training step."""
        self.step_count += 1

        if self.step_count % self.check_frequency == 0:
            issues = self.check_for_nans(loss, inputs, outputs)

            if issues:
                print(f"🚨 NaN detected at step {self.step_count}:")
                for issue in issues:
                    print(f"   - {issue}")

                # Emergency actions
                self._emergency_nan_handling(optimizer)

            # Additional checks
            self._check_gradient_norms()
            self._check_learning_rate(optimizer)

    def _emergency_nan_handling(self, optimizer):
        """Emergency actions when NaN is detected."""
        print("🔧 Applying emergency NaN handling...")

        # Zero out NaN gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.where(
                    torch.isnan(param.grad),
                    torch.zeros_like(param.grad),
                    param.grad
                )

        # Reset optimizer state if needed
        for group in optimizer.param_groups:
            group['lr'] *= 0.5  # Halve learning rate

        print(f"   - Zeroed NaN gradients")
        print(f"   - Halved learning rate to {optimizer.param_groups[0]['lr']}")

    def _check_gradient_norms(self):
        """Check gradient norms for explosion."""
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if total_norm > 100:  # Gradient explosion threshold
            print(f"⚠️ Large gradient norm detected: {total_norm:.2f}")

    def _check_learning_rate(self, optimizer):
        """Check if learning rate is appropriate."""
        lr = optimizer.param_groups[0]['lr']
        if lr > 1e-2:
            print(f"⚠️ Learning rate might be too high: {lr}")
        elif lr < 1e-6:
            print(f"⚠️ Learning rate might be too low: {lr}")

# Usage in training loop
nan_debugger = NaNDebugger(model, check_frequency=10)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        # Debug before backward pass
        nan_debugger.debug_step(loss, data, output, optimizer)

        if not torch.isnan(loss):
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
        else:
            print(f"Skipping step {batch_idx} due to NaN loss")
```

#### **Error**: `Out of memory during training`
```python
# Memory management for training
class MemoryManager:
    """Manage memory usage during training."""

    def __init__(self, device):
        self.device = device
        self.memory_history = []

    def get_memory_usage(self):
        """Get current memory usage."""
        if hasattr(torch.xla.core.xla_model, 'get_memory_info'):
            return torch.xla.core.xla_model.get_memory_info(self.device)
        else:
            # Fallback for older versions
            return {"allocated": 0, "reserved": 0}

    def optimize_memory(self, model, optimizer):
        """Apply memory optimizations."""

        # Gradient accumulation
        accumulation_steps = 4

        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if hasattr(torch.cuda.amp, 'GradScaler') else None

        # Gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        return accumulation_steps, scaler

    def memory_efficient_training_step(self, model, data, target, optimizer,
                                     criterion, accumulation_steps, scaler=None):
        """Memory-efficient training step."""

        # Reduce batch size if needed
        batch_size = data.size(0)
        if batch_size > 16:  # Threshold for memory issues
            # Process in smaller chunks
            chunk_size = 8
            total_loss = 0

            for i in range(0, batch_size, chunk_size):
                chunk_data = data[i:i+chunk_size]
                chunk_target = target[i:i+chunk_size]

                if scaler:
                    with torch.cuda.amp.autocast():
                        chunk_output = model(chunk_data)
                        chunk_loss = criterion(chunk_output, chunk_target)

                    scaler.scale(chunk_loss).backward()
                else:
                    chunk_output = model(chunk_data)
                    chunk_loss = criterion(chunk_output, chunk_target)
                    chunk_loss.backward()

                total_loss += chunk_loss.item()

                # Clear intermediate computations
                del chunk_output, chunk_loss
                torch.xla.core.xla_model.wait_device_ops()

            return total_loss / (batch_size // chunk_size)

        else:
            # Normal processing for smaller batches
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

            return loss.item()

# Usage example
memory_manager = MemoryManager(device)
accumulation_steps, scaler = memory_manager.optimize_memory(model, optimizer)

for batch_idx, (data, target) in enumerate(dataloader):
    if batch_idx % accumulation_steps == 0:
        optimizer.zero_grad()

    loss = memory_manager.memory_efficient_training_step(
        model, data, target, optimizer, criterion, accumulation_steps, scaler
    )

    if (batch_idx + 1) % accumulation_steps == 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    # Memory monitoring
    if batch_idx % 50 == 0:
        memory_info = memory_manager.get_memory_usage()
        print(f"Step {batch_idx}, Memory: {memory_info}")
```

### 4. Inference and Deployment Issues

#### **Error**: `Model serving latency too high`
```python
# Inference optimization and latency debugging
class LatencyOptimizer:
    """Optimize and debug inference latency."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.latency_history = []

    def benchmark_inference(self, example_input, warmup_runs=10, test_runs=100):
        """Benchmark inference performance."""

        print("🔥 Warming up model...")
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(example_input)
                torch.xla.core.xla_model.wait_device_ops()

        print("⏱️ Running latency benchmark...")
        # Actual benchmark
        latencies = []

        for i in range(test_runs):
            start_time = time.time()

            with torch.no_grad():
                output = self.model(example_input)
                torch.xla.core.xla_model.wait_device_ops()

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if i % 20 == 0:
                print(f"   Run {i+1}/{test_runs}: {latency_ms:.2f}ms")

        # Statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        results = {
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_per_second": 1000 / avg_latency,
            "latency_distribution": latencies
        }

        self._print_latency_analysis(results)
        return results

    def _print_latency_analysis(self, results):
        """Print detailed latency analysis."""
        print(f"\n📊 LATENCY ANALYSIS RESULTS")
        print(f"=" * 50)
        print(f"Average Latency: {results['average_latency_ms']:.2f}ms")
        print(f"P95 Latency: {results['p95_latency_ms']:.2f}ms")
        print(f"P99 Latency: {results['p99_latency_ms']:.2f}ms")
        print(f"Throughput: {results['throughput_per_second']:.1f} requests/sec")

        # Performance assessment
        if results['average_latency_ms'] < 50:
            print("✅ Excellent latency performance")
        elif results['average_latency_ms'] < 100:
            print("✅ Good latency performance")
        elif results['average_latency_ms'] < 200:
            print("⚠️ Moderate latency - consider optimization")
        else:
            print("❌ High latency - optimization required")

    def optimize_for_latency(self, batch_sizes=[1, 2, 4, 8, 16]):
        """Find optimal batch size for latency."""

        print("🔍 Finding optimal batch size...")
        results = {}

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            # Create test input
            if hasattr(self.model, 'example_input'):
                example_input = self.model.example_input[:batch_size]
            else:
                # Generic input shape
                example_input = torch.randn(batch_size, 3, 224, 224).to(self.device)

            try:
                benchmark_results = self.benchmark_inference(example_input, warmup_runs=5, test_runs=20)

                per_sample_latency = benchmark_results['average_latency_ms'] / batch_size
                results[batch_size] = {
                    'total_latency_ms': benchmark_results['average_latency_ms'],
                    'per_sample_latency_ms': per_sample_latency,
                    'throughput_per_second': benchmark_results['throughput_per_second'] * batch_size
                }

            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {e}")
                results[batch_size] = {'error': str(e)}

        # Find optimal batch size
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if valid_results:
            optimal_batch_size = min(valid_results.keys(),
                                   key=lambda x: valid_results[x]['per_sample_latency_ms'])

            print(f"\n🎯 OPTIMAL CONFIGURATION")
            print(f"Optimal batch size: {optimal_batch_size}")
            print(f"Per-sample latency: {valid_results[optimal_batch_size]['per_sample_latency_ms']:.2f}ms")
            print(f"Total throughput: {valid_results[optimal_batch_size]['throughput_per_second']:.1f} samples/sec")

        return results

# Usage
latency_optimizer = LatencyOptimizer(compiled_model, device)

# Benchmark current performance
results = latency_optimizer.benchmark_inference(example_input)

# Find optimal batch size
optimization_results = latency_optimizer.optimize_for_latency()
```

#### **Error**: `Model accuracy degradation in production`
```python
# Production accuracy monitoring and debugging
class AccuracyMonitor:
    """Monitor and debug accuracy issues in production."""

    def __init__(self, reference_model, production_model, validation_dataset):
        self.reference_model = reference_model
        self.production_model = production_model
        self.validation_dataset = validation_dataset
        self.accuracy_history = []

    def compare_model_outputs(self, num_samples=100):
        """Compare outputs between reference and production models."""

        differences = []
        dataloader = DataLoader(self.validation_dataset, batch_size=1, shuffle=True)

        self.reference_model.eval()
        self.production_model.eval()

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                if i >= num_samples:
                    break

                # Get predictions from both models
                ref_output = self.reference_model(data)
                prod_output = self.production_model(data)

                # Calculate difference
                diff = torch.abs(ref_output - prod_output).mean().item()
                differences.append(diff)

                if i % 20 == 0:
                    print(f"Compared {i+1}/{num_samples} samples...")

        # Analysis
        avg_diff = np.mean(differences)
        max_diff = np.max(differences)

        print(f"\n📊 MODEL COMPARISON RESULTS")
        print(f"Average output difference: {avg_diff:.6f}")
        print(f"Maximum output difference: {max_diff:.6f}")

        if avg_diff < 1e-4:
            print("✅ Models are numerically equivalent")
        elif avg_diff < 1e-2:
            print("⚠️ Small differences detected - monitor closely")
        else:
            print("❌ Significant differences detected - investigation required")

        return differences

    def validate_data_pipeline(self, production_data_loader):
        """Validate data preprocessing pipeline."""

        print("🔍 Validating data pipeline...")

        issues = []

        # Check data statistics
        sample_batch = next(iter(production_data_loader))
        data, targets = sample_batch

        # Data range checks
        data_min, data_max = data.min().item(), data.max().item()
        if data_min < -5 or data_max > 5:
            issues.append(f"Data range unusual: [{data_min:.2f}, {data_max:.2f}]")

        # Data type checks
        if data.dtype != torch.float32:
            issues.append(f"Data type: {data.dtype} (expected: torch.float32)")

        # Shape consistency
        expected_shape = (data.size(0), 3, 224, 224)  # Example expected shape
        if data.shape[1:] != expected_shape[1:]:
            issues.append(f"Shape mismatch: {data.shape} (expected: {expected_shape})")

        # NaN/Inf checks
        if torch.isnan(data).any():
            issues.append("Data contains NaN values")
        if torch.isinf(data).any():
            issues.append("Data contains infinite values")

        # Target validation
        if targets.dtype not in [torch.long, torch.int64]:
            issues.append(f"Target type: {targets.dtype} (expected: torch.long)")

        if len(issues) == 0:
            print("✅ Data pipeline validation passed")
        else:
            print("❌ Data pipeline issues found:")
            for issue in issues:
                print(f"   - {issue}")

        return issues

    def diagnose_accuracy_drop(self, current_accuracy, baseline_accuracy, threshold=0.05):
        """Diagnose accuracy drop issues."""

        accuracy_drop = baseline_accuracy - current_accuracy

        if accuracy_drop < threshold:
            print(f"✅ Accuracy within acceptable range (drop: {accuracy_drop:.3f})")
            return []

        print(f"❌ Significant accuracy drop detected: {accuracy_drop:.3f}")

        # Systematic diagnosis
        potential_causes = []

        # 1. Model compilation issues
        model_diff = self.compare_model_outputs(num_samples=50)
        if np.mean(model_diff) > 1e-3:
            potential_causes.append("Model compilation or optimization differences")

        # 2. Data pipeline issues
        # (Would need production data loader)
        potential_causes.append("Data preprocessing pipeline changes")

        # 3. Environment differences
        potential_causes.append("Runtime environment differences")

        # 4. Version mismatches
        potential_causes.append("Software version mismatches")

        print(f"\n🔬 POTENTIAL CAUSES:")
        for i, cause in enumerate(potential_causes, 1):
            print(f"   {i}. {cause}")

        return potential_causes

# Usage example
accuracy_monitor = AccuracyMonitor(
    reference_model=original_model,
    production_model=compiled_model,
    validation_dataset=val_dataset
)

# Compare model outputs
differences = accuracy_monitor.compare_model_outputs()

# Validate data pipeline
# data_issues = accuracy_monitor.validate_data_pipeline(production_dataloader)

# Diagnose accuracy issues
# causes = accuracy_monitor.diagnose_accuracy_drop(
#     current_accuracy=0.82,
#     baseline_accuracy=0.89
# )
```

### 5. AWS Infrastructure Issues

#### **Error**: `Unable to launch Trainium/Inferentia instances`
```bash
# Instance availability and quota checker
#!/bin/bash

check_instance_availability() {
    local instance_type=$1
    local region=$2

    echo "🔍 Checking $instance_type availability in $region..."

    # Check service quotas
    aws service-quotas get-service-quota \
        --service-code ec2 \
        --quota-code "L-DB2E81BA" \
        --region $region \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "Quota check failed"

    # Check spot price history (indicates availability)
    aws ec2 describe-spot-price-history \
        --instance-types $instance_type \
        --product-descriptions "Linux/UNIX" \
        --region $region \
        --max-items 1 \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "Spot price unavailable"

    # Try to describe instance types
    aws ec2 describe-instance-types \
        --instance-types $instance_type \
        --region $region \
        --query 'InstanceTypes[0].InstanceType' \
        --output text 2>/dev/null && echo "✅ Instance type available" || echo "❌ Instance type not available"
}

# Check multiple regions for availability
regions=("us-east-1" "us-west-2" "eu-west-1")
instance_types=("trn1.2xlarge" "trn1.32xlarge" "inf2.xlarge" "inf2.8xlarge")

for region in "${regions[@]}"; do
    echo "Region: $region"
    for instance_type in "${instance_types[@]}"; do
        check_instance_availability $instance_type $region
    done
    echo ""
done
```

#### **Error**: `Insufficient IAM permissions for Neuron`
```bash
# IAM permission checker and fixer
#!/bin/bash

check_neuron_permissions() {
    echo "🔐 Checking IAM permissions for Neuron..."

    # Required permissions for Neuron
    required_permissions=(
        "ec2:DescribeInstances"
        "ec2:DescribeInstanceTypes"
        "ec2:DescribeImages"
        "s3:GetObject"
        "s3:PutObject"
        "logs:CreateLogGroup"
        "logs:CreateLogStream"
        "logs:PutLogEvents"
        "cloudwatch:PutMetricData"
    )

    # Test each permission
    for permission in "${required_permissions[@]}"; do
        service=$(echo $permission | cut -d: -f1)
        action=$(echo $permission | cut -d: -f2)

        case $service in
            "ec2")
                aws ec2 describe-instances --max-items 1 >/dev/null 2>&1 && echo "✅ $permission" || echo "❌ $permission"
                ;;
            "s3")
                aws s3 ls >/dev/null 2>&1 && echo "✅ $permission" || echo "❌ $permission"
                ;;
            "logs")
                aws logs describe-log-groups --max-items 1 >/dev/null 2>&1 && echo "✅ $permission" || echo "❌ $permission"
                ;;
            "cloudwatch")
                aws cloudwatch list-metrics --max-items 1 >/dev/null 2>&1 && echo "✅ $permission" || echo "❌ $permission"
                ;;
        esac
    done
}

# Create minimal IAM policy for Neuron
create_neuron_policy() {
    cat << 'EOF' > neuron-minimal-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceTypes",
                "ec2:DescribeImages",
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    echo "📄 Created neuron-minimal-policy.json"
    echo "Apply with: aws iam create-policy --policy-name NeuronMinimalPolicy --policy-document file://neuron-minimal-policy.json"
}

check_neuron_permissions
create_neuron_policy
```

## 🛠️ Advanced Debugging Techniques

### 1. Neuron-Specific Debugging Tools

```python
# Comprehensive Neuron debugging toolkit
class NeuronDebugger:
    """Advanced debugging for Neuron-specific issues."""

    def __init__(self):
        self.debug_info = {}

    def collect_system_info(self):
        """Collect comprehensive system information."""

        import subprocess
        import sys
        import platform

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "neuron_devices": self._get_neuron_devices(),
            "neuron_runtime_version": self._get_neuron_runtime_version(),
            "memory_info": self._get_memory_info(),
            "process_info": self._get_process_info()
        }

        self.debug_info.update(info)
        return info

    def _get_neuron_devices(self):
        """Get Neuron device information."""
        try:
            result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            return f"Error getting device info: {e}"

    def _get_neuron_runtime_version(self):
        """Get Neuron runtime version."""
        try:
            import torch_neuronx
            return torch_neuronx.__version__
        except Exception as e:
            return f"Error getting runtime version: {e}"

    def _get_memory_info(self):
        """Get memory usage information."""
        try:
            result = subprocess.run(['neuron-top', '-n', '1'], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            return f"Error getting memory info: {e}"

    def _get_process_info(self):
        """Get Neuron process information."""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            neuron_processes = [line for line in result.stdout.split('\n') if 'neuron' in line.lower()]
            return '\n'.join(neuron_processes)
        except Exception as e:
            return f"Error getting process info: {e}"

    def diagnose_compilation_failure(self, model, example_input, error_msg):
        """Diagnose model compilation failures."""

        print("🔬 COMPILATION FAILURE DIAGNOSIS")
        print("=" * 50)

        # Analyze error message
        error_analysis = self._analyze_error_message(error_msg)
        print(f"Error Type: {error_analysis['type']}")
        print(f"Likely Cause: {error_analysis['cause']}")
        print(f"Suggested Fix: {error_analysis['fix']}")

        # Model analysis
        model_analysis = self._analyze_model_structure(model)
        print(f"\nModel Analysis:")
        print(f"   Parameters: {model_analysis['param_count']:,}")
        print(f"   Layers: {model_analysis['layer_count']}")
        print(f"   Unsupported ops: {model_analysis['unsupported_ops']}")

        # Input analysis
        input_analysis = self._analyze_input_tensor(example_input)
        print(f"\nInput Analysis:")
        print(f"   Shape: {input_analysis['shape']}")
        print(f"   Data type: {input_analysis['dtype']}")
        print(f"   Memory usage: {input_analysis['memory_mb']:.1f} MB")

        # Recommendations
        recommendations = self._generate_compilation_recommendations(
            error_analysis, model_analysis, input_analysis
        )

        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        return {
            "error_analysis": error_analysis,
            "model_analysis": model_analysis,
            "input_analysis": input_analysis,
            "recommendations": recommendations
        }

    def _analyze_error_message(self, error_msg):
        """Analyze compilation error message."""

        error_patterns = {
            "memory": {
                "patterns": ["out of memory", "memory allocation", "insufficient memory"],
                "cause": "Model too large for available memory",
                "fix": "Reduce batch size or model size"
            },
            "unsupported_op": {
                "patterns": ["unsupported operation", "not supported", "unimplemented"],
                "cause": "Model contains operations not supported by Neuron",
                "fix": "Replace unsupported operations or use CPU fallback"
            },
            "shape_mismatch": {
                "patterns": ["shape mismatch", "dimension mismatch", "size mismatch"],
                "cause": "Dynamic shapes or incompatible tensor dimensions",
                "fix": "Use fixed shapes and compatible dimensions"
            },
            "timeout": {
                "patterns": ["timeout", "compilation timeout", "exceeded time limit"],
                "cause": "Model compilation taking too long",
                "fix": "Reduce model complexity or increase timeout"
            }
        }

        error_msg_lower = error_msg.lower()

        for error_type, info in error_patterns.items():
            if any(pattern in error_msg_lower for pattern in info["patterns"]):
                return {
                    "type": error_type,
                    "cause": info["cause"],
                    "fix": info["fix"]
                }

        return {
            "type": "unknown",
            "cause": "Unrecognized error pattern",
            "fix": "Check Neuron documentation or contact support"
        }

    def _analyze_model_structure(self, model):
        """Analyze model structure for potential issues."""

        param_count = sum(p.numel() for p in model.parameters())
        layer_count = len(list(model.modules()))

        # Check for unsupported operations
        unsupported_ops = []
        model_str = str(model).lower()

        unsupported_patterns = [
            "interpolate", "nonzero", "unique", "topk", "nms"
        ]

        for pattern in unsupported_patterns:
            if pattern in model_str:
                unsupported_ops.append(pattern)

        return {
            "param_count": param_count,
            "layer_count": layer_count,
            "unsupported_ops": unsupported_ops
        }

    def _analyze_input_tensor(self, tensor):
        """Analyze input tensor characteristics."""

        memory_mb = tensor.numel() * tensor.element_size() / 1024 / 1024

        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "memory_mb": memory_mb,
            "device": str(tensor.device)
        }

    def _generate_compilation_recommendations(self, error_analysis, model_analysis, input_analysis):
        """Generate specific recommendations based on analysis."""

        recommendations = []

        # Error-specific recommendations
        if error_analysis["type"] == "memory":
            recommendations.append("Reduce batch size to 1 or smaller")
            recommendations.append("Use gradient checkpointing if training")
            recommendations.append("Consider model pruning or distillation")

        elif error_analysis["type"] == "unsupported_op":
            recommendations.append("Replace unsupported operations with Neuron-compatible alternatives")
            recommendations.append("Use torch.jit.script to identify specific unsupported ops")
            recommendations.append("Consider using CPU fallback for specific layers")

        # Model-specific recommendations
        if model_analysis["param_count"] > 1e9:  # > 1B parameters
            recommendations.append("Model is very large - consider using smaller variant")

        if model_analysis["unsupported_ops"]:
            recommendations.append(f"Address unsupported operations: {', '.join(model_analysis['unsupported_ops'])}")

        # Input-specific recommendations
        if input_analysis["memory_mb"] > 100:  # > 100MB input
            recommendations.append("Input tensor is large - consider reducing input size")

        if len(input_analysis["shape"]) > 4:
            recommendations.append("High-dimensional input may cause issues - consider reshaping")

        return recommendations

    def performance_profiling(self, model, example_input, num_runs=10):
        """Profile model performance in detail."""

        print("📊 PERFORMANCE PROFILING")
        print("=" * 30)

        # Compilation profiling
        compilation_start = time.time()
        try:
            compiled_model = torch_neuronx.trace(model, example_input)
            compilation_time = time.time() - compilation_start
            print(f"✅ Compilation successful: {compilation_time:.2f}s")
        except Exception as e:
            compilation_time = time.time() - compilation_start
            print(f"❌ Compilation failed after {compilation_time:.2f}s: {e}")
            return None

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = compiled_model(example_input)
                torch.xla.core.xla_model.wait_device_ops()

        # Performance measurement
        print(f"Running {num_runs} inference iterations...")
        latencies = []

        for i in range(num_runs):
            start_time = time.time()

            with torch.no_grad():
                output = compiled_model(example_input)
                torch.xla.core.xla_model.wait_device_ops()

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        # Statistics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        print(f"\n📈 PERFORMANCE RESULTS:")
        print(f"   Compilation time: {compilation_time:.2f}s")
        print(f"   Average latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"   Min/Max latency: {min_latency:.2f}/{max_latency:.2f} ms")
        print(f"   Throughput: {1000/avg_latency:.1f} inferences/sec")

        return {
            "compilation_time": compilation_time,
            "average_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "throughput_per_sec": 1000/avg_latency
        }

# Usage example
debugger = NeuronDebugger()

# Collect system information
system_info = debugger.collect_system_info()
print("System Information:", json.dumps(system_info, indent=2))

# Diagnose compilation failure
try:
    compiled_model = torch_neuronx.trace(model, example_input)
except Exception as e:
    diagnosis = debugger.diagnose_compilation_failure(model, example_input, str(e))

# Performance profiling
if 'compiled_model' in locals():
    perf_results = debugger.performance_profiling(compiled_model, example_input)
```

### 2. Logging and Monitoring Setup

```python
# Comprehensive logging setup for Neuron debugging
import logging
import sys
from datetime import datetime

def setup_neuron_logging(log_level=logging.INFO, log_file=None):
    """Setup comprehensive logging for Neuron debugging."""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Setup file handler if requested
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(log_level)

    # Clear existing handlers
    main_logger.handlers.clear()

    # Add new handlers
    for handler in handlers:
        main_logger.addHandler(handler)

    # Configure Neuron-specific loggers
    neuron_loggers = [
        'torch_neuronx',
        'torch_xla',
        'neuronx_distributed',
        'neuron_compiler'
    ]

    for logger_name in neuron_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

    logging.info("🔧 Neuron logging configured")
    return main_logger

# Usage
logger = setup_neuron_logging(
    log_level=logging.DEBUG,
    log_file=f"neuron_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
```

## 📚 Reference Quick Cards

### Neuron Error Codes Reference
| Error Code | Meaning | Common Causes | Quick Fix |
|------------|---------|---------------|-----------|
| **NEURON_RT_EXEC_ERROR** | Runtime execution error | Model mismatch, memory issues | Recompile model |
| **NEURON_RT_TIMEOUT** | Operation timeout | Large model, insufficient resources | Reduce model size |
| **NEURON_RT_INVALID_HANDLE** | Invalid resource handle | Memory corruption, driver issues | Restart Neuron runtime |
| **NEURON_RT_OUT_OF_MEMORY** | Memory exhaustion | Model too large, memory leaks | Reduce batch size |

### Environment Variables for Debugging
```bash
# Enable maximum verbosity
export NEURON_RT_LOG_LEVEL=DEBUG
export NEURONX_COMPILE_CACHE_URL=/tmp/neuron_cache
export NEURON_FRAMEWORK_DEBUG=1

# Memory debugging
export NEURON_RT_VISIBLE_CORES=0,1  # Limit cores for debugging
export MALLOC_CHECK_=1  # Enable memory debugging

# Performance debugging
export NEURON_PROFILE=profile.json
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
```

### Emergency Commands
```bash
# Emergency cleanup
sudo systemctl restart neuron-monitor
sudo systemctl restart neuron-discovery
sudo pkill -f neuron

# Memory cleanup
echo 3 | sudo tee /proc/sys/vm/drop_caches
sync

# Reset Neuron devices
sudo modprobe -r neuron
sudo modprobe neuron

# Check system health
neuron-ls -v
neuron-top -n 1
dmesg | grep -i neuron | tail -20
```

This comprehensive guide covers the most common issues and provides systematic approaches to debugging. The tools and techniques should help you quickly identify and resolve problems when working with AWS Trainium and Inferentia.

Remember to always check the version matrix first, as many issues are version-specific and can be resolved by using tested compatible versions.
