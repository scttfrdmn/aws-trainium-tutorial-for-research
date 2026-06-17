# Video Tutorial 1: Getting Started with AWS Trainium & Inferentia

**Duration**: 15 minutes  
**Difficulty**: Beginner  
**Prerequisites**: AWS account, basic Linux knowledge

## 🎯 Learning Objectives

By the end of this tutorial, viewers will be able to:
1. Launch a Trainium instance on AWS
2. Install the Neuron SDK and torch-neuronx
3. Compile and run their first model on Neuron hardware
4. Understand basic cost monitoring

## 📋 Tutorial Script

### Opening (0:00 - 1:00)
**[TITLE CARD: Getting Started with AWS Trainium & Inferentia]**

**Narrator**: "Welcome to our comprehensive tutorial series on AWS Trainium and Inferentia for research. I'm [Name], and in this first video, we'll get you up and running with your first Neuron-optimized model in just 15 minutes.

By the end of this tutorial, you'll have a working Trainium environment and understand the basic workflow. Let's dive in!"

**[SHOW: Tutorial overview slide with key steps]**

### Section 1: AWS Setup (1:00 - 4:00)

#### AWS Console Login (1:00 - 1:30)
**[SCREEN: AWS Console homepage]**

**Narrator**: "First, let's log into the AWS console. I'm already logged in here. If you don't have an AWS account, you'll need to create one - there are links in the description for the free tier.

Notice that I'm in the us-west-2 region. Trainium instances are available in specific regions, so make sure you're in us-west-2, us-east-1, or one of the other supported regions."

**[HIGHLIGHT: Region selector in top navigation]**

#### Instance Launch (1:30 - 3:30)
**[SCREEN: EC2 Dashboard]**

**Narrator**: "Now let's launch our first Trainium instance. I'll navigate to EC2, and click 'Launch Instance'.

For the AMI, I'm selecting the Deep Learning AMI - Ubuntu. This comes pre-configured with many ML frameworks and makes our setup much easier."

**[SHOW: AMI selection screen, highlight Deep Learning AMI]**

**Narrator**: "For instance type, we want a trn1 instance. I'm selecting trn1.2xlarge for this tutorial - it has 2 Neuron cores and is perfect for learning. In production, you might want trn1.32xlarge for maximum performance."

**[SHOW: Instance type selection, highlight trn1.2xlarge]**

**Narrator**: "I'll create a new key pair for SSH access. Make sure to download this .pem file - you'll need it to connect to your instance."

**[SHOW: Key pair creation dialog]**

#### Cost Monitoring Setup (3:30 - 4:00)
**Narrator**: "Before we launch, let's set up a billing alert. This is crucial for research - you don't want any surprises on your AWS bill. I'll navigate to Billing, then Budgets, and create a simple budget for $50. You can adjust this based on your needs."

**[SHOW: AWS Budgets creation screen]**

### Section 2: Environment Setup (4:00 - 8:00)

#### SSH Connection (4:00 - 4:30)
**[SCREEN: Terminal window]**

**Narrator**: "Great! Our instance is now running. Let's connect via SSH. I'll use the command shown in the EC2 console."

```bash
ssh -i ~/Downloads/my-key.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com
```

**[SHOW: SSH connection establishing]**

#### System Verification (4:30 - 5:30)
**Narrator**: "First, let's verify our system has Neuron hardware. I'll run the neuron-ls command."

```bash
neuron-ls
```

**[SHOW: Command output showing Neuron devices]**

**Narrator**: "Perfect! We can see 2 Neuron cores available. Now let's check our system information and update the package manager."

```bash
# Check instance type
curl -s http://169.254.169.254/latest/meta-data/instance-type

# Update system
sudo apt update
```

#### Neuron SDK Installation (5:30 - 7:00)
**Narrator**: "The Deep Learning AMI includes some Neuron components, but let's make sure we have the latest. I'll install the Neuron SDK and torch-neuronx."

```bash
# Install torch-neuronx
pip install torch-neuronx==2.2.0 --extra-index-url https://pip.repos.neuron.amazonaws.com

# Verify installation
python -c "import torch_neuronx; print('torch-neuronx version:', torch_neuronx.__version__)"
```

**[SHOW: Installation progress and verification]**

#### Environment Verification (7:00 - 8:00)
**Narrator**: "Let's verify everything is working correctly by checking our XLA devices."

```bash
python -c "
import torch_xla.core.xla_model as xm
print('Available XLA devices:', xm.get_xla_supported_devices())
print('Current device:', xm.xla_device())
"
```

**[SHOW: Python output showing XLA Neuron devices]**

### Section 3: First Model (8:00 - 12:00)

#### Simple Model Creation (8:00 - 9:30)
**[SCREEN: Text editor with Python code]**

**Narrator**: "Now for the exciting part - let's compile and run our first model on Neuron. I'll create a simple neural network for demonstration."

```python
#!/usr/bin/env python3
"""First Neuron Model Example"""

import torch
import torch.nn as nn
import torch_neuronx
import torch_xla.core.xla_model as xm

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create model and example input
model = SimpleNN()
example_input = torch.randn(1, 784)

print("Model created successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**[SHOW: Code being typed and executed]**

#### Model Compilation (9:30 - 11:00)
**Narrator**: "Now let's compile this model for Neuron. This is where the magic happens - torch-neuronx will optimize our model specifically for Trainium hardware."

```python
# Compile model for Neuron
print("Compiling model for Neuron...")
compiled_model = torch_neuronx.trace(model, example_input)
print("Compilation successful!")

# Test the compiled model
print("Running inference...")
with torch.no_grad():
    output = compiled_model(example_input)
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0][:5]}")
```

**[SHOW: Compilation progress and output]**

**Narrator**: "Notice that compilation took a few seconds. This is normal - Neuron is analyzing and optimizing your model. In production, you'd typically compile once and reuse the compiled model."

#### Performance Comparison (11:00 - 12:00)
**Narrator**: "Let's do a quick performance comparison between CPU and Neuron execution."

```python
import time

# CPU timing
model_cpu = SimpleNN()
input_cpu = torch.randn(32, 784)  # Larger batch for better measurement

start_time = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model_cpu(input_cpu)
cpu_time = time.time() - start_time

# Neuron timing  
input_neuron = torch.randn(32, 784)
compiled_batch_model = torch_neuronx.trace(model, input_neuron)

start_time = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = compiled_batch_model(input_neuron)
neuron_time = time.time() - start_time

print(f"CPU time: {cpu_time:.3f}s")
print(f"Neuron time: {neuron_time:.3f}s")
print(f"Speedup: {cpu_time/neuron_time:.1f}x")
```

**[SHOW: Timing results]**

### Section 4: Monitoring and Cleanup (12:00 - 14:00)

#### Resource Monitoring (12:00 - 13:00)
**Narrator**: "Let's monitor our Neuron usage with the built-in monitoring tools."

```bash
# Monitor Neuron utilization
neuron-monitor &
sleep 5

# Run some inference to see activity
python -c "
import torch
import torch_neuronx
model = torch.nn.Linear(100, 10)
input_data = torch.randn(8, 100)
traced_model = torch_neuronx.trace(model, input_data)
for i in range(20):
    _ = traced_model(input_data)
"

# Stop monitoring
pkill neuron-monitor
```

**[SHOW: neuron-monitor output with utilization graphs]**

#### Cost Monitoring (13:00 - 13:30)
**Narrator**: "Always keep an eye on your costs. A trn1.2xlarge costs about $1.34 per hour. For this 15-minute tutorial, we've spent roughly $0.34. Remember to terminate your instance when you're done experimenting!"

**[SHOW: AWS Cost Explorer or billing dashboard]**

#### Instance Cleanup (13:30 - 14:00)
**Narrator**: "Finally, let's clean up. I'll terminate this instance to avoid ongoing charges. In the EC2 console, I'll select my instance and choose 'Terminate'."

**[SHOW: Instance termination in AWS console]**

**Narrator**: "Always double-check that your instances are terminated. You can also set up auto-termination using scripts or AWS Lambda for additional protection."

### Closing (14:00 - 15:00)
**Narrator**: "Congratulations! You've successfully:
- Launched your first Trainium instance
- Installed the Neuron SDK
- Compiled and run a model on Neuron hardware
- Learned basic monitoring and cost management

In our next tutorial, we'll dive deeper into training workflows and explore real research applications. Don't forget to check out the code examples in the GitHub repository linked in the description.

Thanks for watching, and I'll see you in the next video!"

**[SHOW: End screen with links to next tutorial and GitHub repo]**

## 🎬 Production Notes

### Visual Elements
- **Screen Recording**: Capture at 1080p minimum, 4K preferred for code clarity
- **Cursor Highlighting**: Use cursor highlighting for important clicks
- **Code Syntax**: Ensure proper syntax highlighting in all code editors
- **Terminal Styling**: Use a professional terminal theme with good contrast

### Audio Considerations
- **Pace**: Slow enough for technical content, with pauses for comprehension
- **Pronunciation**: Clear pronunciation of technical terms
- **Background**: No background music during code sections
- **Levels**: Consistent audio levels throughout

### Interactive Elements
- **Code Downloads**: All code available in GitHub repository
- **Timestamps**: Video chapters for easy navigation
- **Captions**: Full captions and transcript available
- **Resources**: Links to additional resources in description

## 🔧 Technical Setup

### Demo Environment
```bash
# Pre-tutorial setup script
#!/bin/bash
set -e

echo "Setting up demo environment..."

# Update system
sudo apt update

# Install required packages
pip install torch-neuronx==2.2.0 --extra-index-url https://pip.repos.neuron.amazonaws.com

# Verify setup
python -c "import torch_neuronx; print('Setup complete!')"

echo "Demo environment ready!"
```

### Troubleshooting
Common issues during tutorial recording:
1. **Compilation timeouts**: Increase NEURON_COMPILE_TIMEOUT
2. **SSH connection issues**: Check security group settings
3. **Package conflicts**: Use fresh AMI for each recording
4. **Audio sync**: Record audio and video separately if needed

## 📚 Supplementary Materials

### Required Reading
- AWS Trainium instance types documentation
- Neuron SDK installation guide
- Basic PyTorch tutorial (for beginners)

### Additional Resources
- AWS Free Tier information
- Cost optimization best practices
- Neuron model compilation guide

### Practice Exercises
1. Modify the neural network architecture
2. Try different batch sizes
3. Experiment with different model types
4. Set up cost alerts for your account

---

*This tutorial provides a solid foundation for getting started with AWS Trainium and Inferentia for research applications.*