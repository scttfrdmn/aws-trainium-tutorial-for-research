#!/usr/bin/env python3
"""Train on Trainium → serve on Inferentia: the end-to-end shape of a research deployment.

This is the tutorial's **"train then serve"** example. It shows how the two halves of an ML
project fit together on Neuron hardware: you train on a Trainium instance (the PyTorch/XLA path
the other examples teach), then **compile the trained model once for inference** and serve it on a
cheaper Inferentia instance behind an HTTP endpoint — with cost tracking on both phases.

> **Assumed knowledge:** you've run the [biomedical NER example](../use_cases/biomedical_ner.py)
> (so the XLA training path is familiar) and read the
> [best-practices chapter](../../docs/trainium_development_best_practices.md).
> **What you'll get:** the orchestration shape of a real train→serve pipeline — SSM-resolved Neuron
> AMIs, spot + auto-terminate, the crucial **`trace()`-is-inference-only** boundary between training
> and serving, and where the costs land. Not a model you'll copy verbatim, but a template you adapt.

## ⚠️ Illustrative orchestration — read it, adapt it, don't run it blind

Unlike the validated single-file examples, this one **launches real, billable EC2 instances** and
serves an **unauthenticated** Flask endpoint, and it trains on a **placeholder dataset** (a CSV you
supply). It is deliberately *not* in the validation harness (the harness validates single-device
`run(config)` examples; this orchestrates a multi-instance workflow). Treat every cost/latency
number it prints as an **estimate**, and swap in your own dataset + governance before any real use.

## The one lesson to take away: training graph ≠ inference graph

The training loop runs on the **XLA lazy-tensor** path (`xm.xla_device()`, `xm.optimizer_step()`,
`xm.mark_step()`). When training finishes, the model is **`torch_neuronx.trace()`-d once** into a
*frozen inference graph* and saved — you serve that. You cannot backprop through a traced graph;
`trace()` is for inference only. Getting this boundary right is the whole point of the example.

## Platform note (June 2026)

AWS positions **Trainium2 for both training and inference**, and NxD Inference dropped Inf2/Trn1
support in Neuron 2.29. This example serves on **Inf2** for simplicity and to show the classic
train-Trn/serve-Inf split; for new or large-scale serving, prefer **Trn2 + NxD Inference + the vLLM
plugin**. See [`../../VERSION_MATRIX.md`](../../VERSION_MATRIX.md) for the decision guide.

## Cost numbers are illustrative

The `$/hr` rates live in one clearly-labeled table (`ILLUSTRATIVE_HOURLY_USD` below) and are rough,
region- and time-dependent estimates used to demonstrate the comparison *logic* — not quotes, and
not measured results from a controlled run. Always confirm current pricing at
https://aws.amazon.com/ec2/pricing/ before budgeting.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import boto3

# Illustrative on-demand/spot $/hr by instance — NOT quotes. One place to edit; every cost figure
# the pipeline prints derives from here so there are no hand-typed magic numbers scattered around.
# Verify against https://aws.amazon.com/ec2/pricing/ before using any of these for a budget.
ILLUSTRATIVE_HOURLY_USD: dict[str, float] = {
    "trn1.2xlarge": 0.40,  # spot, us-east-2-ish
    "trn1.32xlarge": 6.45,
    "trn2.48xlarge": 12.00,
    "inf2.xlarge": 0.227,
    # GPU reference points for the comparison (on-demand, illustrative):
    "p3.2xlarge": 3.06,  # V100 — a common academic baseline
    "p5.48xlarge": 98.0,  # H100 — large-scale baseline
}


class TrainiumToInferentiaPipeline:
    """Complete pipeline from training to deployment with cost tracking"""

    def __init__(self, project_name, s3_bucket):
        self.project_name = project_name
        self.s3_bucket = s3_bucket
        self.train_instance_type = (
            "trn1.32xlarge"  # set in train_on_trainium; used for costing
        )
        self.ec2 = boto3.client("ec2")
        self.s3 = boto3.client("s3")
        self.ssm = boto3.client("ssm")

    def _latest_neuron_dlami(self, pytorch_version="2.9", os_version="ubuntu-24.04"):
        """Resolve the current AWS Neuron Deep Learning AMI id via SSM Parameter Store.

        Hardcoding an AMI id (e.g. ami-0abc...) breaks across regions and over time, and AWS
        publishes the latest DLAMI ids as public SSM parameters. This looks one up at launch
        time so the example actually boots a Neuron-capable image.

        The parameter path is versioned by PyTorch release, e.g. (verified us-west-2, June 2026):
            /aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id
        which currently resolves to "Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)".
        List available paths with:
            aws ssm get-parameters-by-path \\
                --path /aws/service/neuron --recursive --query 'Parameters[].Name'
        """
        param = (
            f"/aws/service/neuron/dlami/pytorch-{pytorch_version}/"
            f"{os_version}/latest/image_id"
        )
        try:
            return self.ssm.get_parameter(Name=param)["Parameter"]["Value"]
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
            raise RuntimeError(
                "Could not resolve the latest Neuron DLAMI from SSM "
                f"(parameter '{param}'). Look up the current parameter name with "
                "`aws ssm get-parameters-by-path --path /aws/service/neuron --recursive` "
                "or pass an explicit AMI id."
            ) from exc

    def train_on_trainium(self, model_class, dataset_path, config):
        """Train model on Trainium with automatic cost tracking"""
        print("🚀 Phase 1: Training on Trainium")
        instance_type = config.get("instance_type", "trn1.32xlarge")
        self.train_instance_type = instance_type

        # Generate training script
        training_script = self._generate_training_script(
            model_class, dataset_path, config
        )

        # Launch Trainium instance
        user_data = f"""#!/bin/bash
# Update system and install dependencies
apt-get update
apt-get install -y python3-pip awscli at

# Install Neuron SDK
pip3 install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com
pip3 install transformers datasets accelerate boto3 numpy

# Set environment variables
export NEURON_CC_FLAGS="--model-type=transformer --enable-saturate-infinity"
export XLA_USE_BF16=1

# Download training script
aws s3 cp s3://{self.s3_bucket}/scripts/train.py /home/ubuntu/train.py
aws s3 cp s3://{dataset_path} /home/ubuntu/dataset.tar.gz

# Extract dataset
cd /home/ubuntu
tar -xzf dataset.tar.gz

# Cost tracking setup
cat > /home/ubuntu/cost_monitor.py << 'EOF'
{self._get_cost_monitor_script(instance_type)}
EOF

# Run training with cost monitoring
nohup python3 /home/ubuntu/cost_monitor.py > /home/ubuntu/cost_log.txt 2>&1 &
python3 /home/ubuntu/train.py

# Upload results
aws s3 sync /home/ubuntu/results s3://{self.s3_bucket}/experiments/{self.project_name}/
aws s3 cp /home/ubuntu/cost_log.txt s3://{self.s3_bucket}/experiments/{self.project_name}/training_costs.txt

# Auto-terminate
sudo shutdown -h now
"""

        # Upload training script to S3
        self.s3.put_object(
            Bucket=self.s3_bucket, Key="scripts/train.py", Body=training_script
        )

        # Launch instance on the current Neuron Deep Learning AMI (resolved at runtime).
        response = self.ec2.run_instances(
            ImageId=self._latest_neuron_dlami(),
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
            IamInstanceProfile={"Name": "ML-Research-EC2-Role"},
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"Training-{self.project_name}"},
                        {"Key": "Phase", "Value": "Training"},
                        {"Key": "Project", "Value": self.project_name},
                    ],
                }
            ],
        )

        instance_id = response["Instances"][0]["InstanceId"]
        print(f"✅ Launched Trainium instance: {instance_id}")

        # Wait for training to complete
        training_time, training_cost = self._wait_for_training_completion(instance_id)

        return {
            "instance_id": instance_id,
            "training_time_hours": training_time,
            "training_cost_usd": training_cost,
            "model_path": f"s3://{self.s3_bucket}/experiments/{self.project_name}/model_inferentia.pt",
        }

    def deploy_on_inferentia(self, model_path, config):
        """Deploy trained model on Inferentia for inference"""
        print("\n🚀 Phase 2: Deploying on Inferentia")

        # Generate inference server script
        inference_script = self._generate_inference_script(model_path, config)

        user_data = f"""#!/bin/bash
# Install dependencies
apt-get update
apt-get install -y python3-pip awscli
pip3 install torch-neuronx --extra-index-url https://pip.repos.neuron.amazonaws.com
pip3 install flask boto3 numpy

# Download model and inference script
aws s3 cp {model_path} /home/ubuntu/model_inferentia.pt
aws s3 cp s3://{self.s3_bucket}/scripts/inference_server.py /home/ubuntu/inference_server.py

# Cost tracking for inference
cat > /home/ubuntu/inference_monitor.py << 'EOF'
{self._get_inference_monitor_script()}
EOF

# Start inference server with monitoring
nohup python3 /home/ubuntu/inference_monitor.py > /home/ubuntu/inference_cost_log.txt 2>&1 &
python3 /home/ubuntu/inference_server.py

# Setup auto-shutdown after configured hours
echo "sudo shutdown -h now" | at now + {config.get("inference_hours", 24)} hours
"""

        # Upload inference script
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key="scripts/inference_server.py",
            Body=inference_script,
        )

        # Launch Inferentia instance on the current Neuron Deep Learning AMI.
        response = self.ec2.run_instances(
            ImageId=self._latest_neuron_dlami(),
            InstanceType="inf2.xlarge",
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            IamInstanceProfile={"Name": "ML-Research-EC2-Role"},
            SecurityGroups=["ML-Research-SG"],
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"Inference-{self.project_name}"},
                        {"Key": "Phase", "Value": "Inference"},
                        {"Key": "Project", "Value": self.project_name},
                    ],
                }
            ],
        )

        instance_id = response["Instances"][0]["InstanceId"]

        # Wait for instance to be running and get public IP
        waiter = self.ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id])

        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = response["Reservations"][0]["Instances"][0]["PublicIpAddress"]

        inf_hourly = ILLUSTRATIVE_HOURLY_USD["inf2.xlarge"]
        print(f"✅ Deployed on Inferentia: {instance_id}")
        print(f"🌐 Inference endpoint: http://{public_ip}:8080/predict")
        print(
            f"💰 Illustrative cost: ${inf_hourly:.3f}/hour (inf2.xlarge spot — verify pricing)"
        )

        return {
            "instance_id": instance_id,
            "endpoint": f"http://{public_ip}:8080/predict",
            "hourly_cost": inf_hourly,
        }

    def _generate_training_script(self, model_class, dataset_path, config):
        """Generate training script for Trainium"""
        return f"""
import torch
import torch_xla.core.xla_model as xm
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import json
import boto3
from datetime import datetime

class ClimateDataset(Dataset):
    def __init__(self, data_path):
        import pandas as pd
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simplified dataset implementation
        return {{
            'text': self.data.iloc[idx]['text'],
            'label': self.data.iloc[idx]['label']
        }}

def train_model():
    print("🔄 Starting training on Trainium...")

    # Setup XLA device. On Trainium, PyTorch training uses the PyTorch/XLA lazy-tensor path:
    # place tensors on the XLA device, and materialize the graph with xm.mark_step().
    # IMPORTANT: do NOT torch_neuronx.trace() the model for training -- trace() produces a
    # frozen inference graph that you cannot backprop through. trace() is for inference only,
    # after training (see the Inferentia compile step at the end).
    device = xm.xla_device()
    print(f"Using device: {{device}}")

    # Load model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels={config.get("num_labels", 2)}
    )

    # Move the trainable model to the XLA (Trainium) device.
    model = model.to(device)

    # Setup training
    dataset = ClimateDataset('dataset/train.csv')
    train_loader = DataLoader(dataset, batch_size={config.get("batch_size", 32)}, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr={config.get("learning_rate", 2e-5)})
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_loss = float('inf')

    for epoch in range({config.get("epochs", 10)}):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            # Tokenize
            encoded = tokenizer(
                batch['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = torch.tensor(batch['label']).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels)

            # Backward pass. xm.optimizer_step() applies grads; xm.mark_step() materializes
            # the lazy XLA graph (one compile/execute step on Trainium).
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {{epoch}}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {{epoch}} completed. Average Loss: {{avg_loss:.4f}}")

        # Save best model (checkpoint only -- compile for Inferentia once, after training).
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({{
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }}, '/home/ubuntu/results/checkpoint.pt')
            print(f"✅ Saved new best checkpoint (loss: {{best_loss:.4f}})")

    print("🎉 Training completed!")

    # --- Compile the trained model for Inferentia (inference) -------------------------------
    # Reload best weights onto CPU, then trace once for inference. Tracing happens on CPU
    # tensors; torch_neuronx.trace() compiles the inference graph for Inf2/Trn2 serving.
    print("🔧 Compiling best model for Inferentia inference...")
    best_ckpt = torch.load('/home/ubuntu/results/checkpoint.pt', map_location='cpu')
    cpu_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels={config.get("num_labels", 2)}
    )
    cpu_model.load_state_dict(best_ckpt['model_state_dict'])
    cpu_model.eval()

    example_inputs = (
        torch.zeros(1, 512, dtype=torch.long),   # input_ids
        torch.ones(1, 512, dtype=torch.long),    # attention_mask
    )
    inference_model = torch_neuronx.trace(cpu_model, example_inputs)
    torch.jit.save(inference_model, '/home/ubuntu/results/model_inferentia.pt')
    print("✅ Saved Inferentia-ready model: model_inferentia.pt")

    # Generate training report
    report = {{
        'project': '{self.project_name}',
        'training_completed': datetime.now().isoformat(),
        'final_loss': best_loss,
        'epochs_trained': {config.get("epochs", 10)},
        'batch_size': {config.get("batch_size", 32)},
        'learning_rate': {config.get("learning_rate", 2e-5)}
    }}

    with open('/home/ubuntu/results/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs('/home/ubuntu/results', exist_ok=True)
    train_model()
"""

    def _generate_inference_script(self, model_path, config):
        """Generate inference server script for Inferentia"""
        return """
from flask import Flask, request, jsonify
import torch
import torch_neuronx
from transformers import AutoTokenizer
import numpy as np
import time
import json
from datetime import datetime

app = Flask(__name__)

# Global variables
model = None
tokenizer = None
request_count = 0
total_latency = 0
start_time = datetime.now()

def load_model():
    global model, tokenizer

    print("📥 Loading model for Inferentia...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load compiled model
    model = torch.jit.load('/home/ubuntu/model_inferentia.pt')
    model.eval()

    print("✅ Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    global request_count, total_latency

    # Use a distinct name -- do NOT shadow the module-level `start_time` (a datetime used for
    # uptime/cost), or the runtime math below raises TypeError (float minus datetime).
    request_start = time.time()

    try:
        # Get input data
        data = request.json
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        # Tokenize
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        # Run inference
        with torch.no_grad():
            outputs = model(encoded['input_ids'], encoded['attention_mask'])
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Calculate latency
        latency_ms = (time.time() - request_start) * 1000

        # Update statistics
        request_count += len(texts)
        total_latency += latency_ms
        avg_latency = total_latency / request_count if request_count > 0 else 0

        # Calculate costs (server uptime since process start, not this request)
        hourly_rate = 0.227  # inf2.xlarge spot price (verify current pricing)
        runtime_hours = (datetime.now() - start_time).total_seconds() / 3600
        total_cost = runtime_hours * hourly_rate
        cost_per_1k_requests = (total_cost / request_count) * 1000 if request_count > 0 else 0

        return jsonify({
            'predictions': predictions.tolist(),
            'latency_ms': round(latency_ms, 2),
            'batch_size': len(texts),
            'statistics': {
                'total_requests': request_count,
                'average_latency_ms': round(avg_latency, 2),
                'total_cost_usd': round(total_cost, 4),
                'cost_per_1k_requests': round(cost_per_1k_requests, 4),
                'requests_per_dollar': round(1000 / cost_per_1k_requests, 0) if cost_per_1k_requests > 0 else 0
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'loaded' if model is not None else 'not_loaded',
        'uptime_hours': round((datetime.now() - start_time).total_seconds() / 3600, 2),
        'total_requests': request_count
    })

@app.route('/stats', methods=['GET'])
def stats():
    runtime_hours = (datetime.now() - start_time).total_seconds() / 3600
    hourly_rate = 0.227
    total_cost = runtime_hours * hourly_rate

    return jsonify({
        'runtime_hours': round(runtime_hours, 2),
        'total_requests': request_count,
        'total_cost_usd': round(total_cost, 4),
        'cost_per_request': round(total_cost / request_count, 6) if request_count > 0 else 0,
        'requests_per_hour': round(request_count / runtime_hours, 0) if runtime_hours > 0 else 0,
        'average_latency_ms': round(total_latency / request_count, 2) if request_count > 0 else 0
    })

if __name__ == '__main__':
    load_model()
    print("🚀 Starting inference server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
"""

    def _get_cost_monitor_script(self, instance_type):
        """Generate cost monitoring script"""
        hourly_rates = {
            "trn1.2xlarge": 0.40,
            "trn1.32xlarge": 6.45,
            "trn2.48xlarge": 12.00,
        }

        return f"""
import time
import psutil
import json
from datetime import datetime

class TrainingCostMonitor:
    def __init__(self):
        self.instance_type = '{instance_type}'
        self.hourly_rate = {hourly_rates.get(instance_type, 10.0)}
        self.start_time = datetime.now()
        self.experiment_name = '{self.project_name}'

    def run(self):
        while True:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            current_cost = elapsed_hours * self.hourly_rate

            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            log_entry = {{
                'timestamp': datetime.now().isoformat(),
                'experiment': self.experiment_name,
                'instance_type': self.instance_type,
                'elapsed_hours': round(elapsed_hours, 2),
                'current_cost_usd': round(current_cost, 2),
                'hourly_rate': self.hourly_rate,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            }}

            print(f"💰 Cost: ${{current_cost:.2f}} | ⏱️  Runtime: {{elapsed_hours:.2f}}h | 🖥️  CPU: {{cpu_percent}}% | 🧠 Memory: {{memory_percent}}%")

            # Save to file
            with open('/home/ubuntu/cost_metrics.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\\n')

            time.sleep(300)  # Log every 5 minutes

monitor = TrainingCostMonitor()
monitor.run()
"""

    def _get_inference_monitor_script(self):
        """Generate inference cost monitoring script"""
        return f"""
import time
import json
from datetime import datetime
import requests

class InferenceCostMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.hourly_rate = 0.227  # inf2.xlarge spot price

    def run(self):
        while True:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            current_cost = elapsed_hours * self.hourly_rate

            # Get stats from inference server
            try:
                response = requests.get('http://localhost:8080/stats', timeout=5)
                stats = response.json() if response.status_code == 200 else {{}}
            except:
                stats = {{}}

            log_entry = {{
                'timestamp': datetime.now().isoformat(),
                'experiment': '{self.project_name}',
                'phase': 'inference',
                'elapsed_hours': round(elapsed_hours, 2),
                'current_cost_usd': round(current_cost, 2),
                'hourly_rate': self.hourly_rate,
                'total_requests': stats.get('total_requests', 0),
                'requests_per_hour': stats.get('requests_per_hour', 0),
                'average_latency_ms': stats.get('average_latency_ms', 0)
            }}

            print(f"💰 Inference Cost: ${{current_cost:.2f}} | ⏱️  Runtime: {{elapsed_hours:.2f}}h | 📊 Requests: {{stats.get('total_requests', 0)}}")

            # Save to file
            with open('/home/ubuntu/inference_cost_metrics.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\\n')

            time.sleep(300)  # Log every 5 minutes

monitor = InferenceCostMonitor()
monitor.run()
"""

    def _wait_for_training_completion(self, instance_id):
        """Wait for training instance to complete and calculate costs"""
        print("⏳ Waiting for training to complete...")

        start_time = time.time()

        while True:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            state = response["Reservations"][0]["Instances"][0]["State"]["Name"]

            rate = ILLUSTRATIVE_HOURLY_USD.get(self.train_instance_type, 10.0)
            if state == "terminated":
                end_time = time.time()
                training_time = (end_time - start_time) / 3600
                training_cost = training_time * rate

                print("✅ Training completed!")
                print(f"⏱️  Training time: {training_time:.2f} hours")
                print(f"💰 Training cost (illustrative): ${training_cost:.2f}")

                return training_time, training_cost

            elif state == "running":
                elapsed = (time.time() - start_time) / 3600
                estimated_cost = elapsed * rate
                print(
                    f"🔄 Training in progress... {elapsed:.1f}h elapsed, ${estimated_cost:.2f} spent"
                )

            time.sleep(300)  # Check every 5 minutes

    def run_cost_comparison(self, training_result, inference_result):
        """Generate an ILLUSTRATIVE cost comparison report.

        Every figure derives from ILLUSTRATIVE_HOURLY_USD (one labeled table) so there are no
        hand-typed savings percentages. The GPU baseline is an explicit instance rate, not a
        made-up multiplier. These are planning aids, not measured results — verify pricing.
        """
        print("\n📊 Cost Comparison Report (ILLUSTRATIVE — verify pricing)")
        print("=" * 50)

        # Training: compare the actual Trainium instance used against a named GPU baseline at the
        # SAME wall-clock hours (a rough "if you ran the same job on a GPU" proxy, not a benchmark).
        train_hours = training_result["training_time_hours"]
        train_cost = training_result["training_cost_usd"]
        gpu_train_rate = ILLUSTRATIVE_HOURLY_USD["p5.48xlarge"]  # H100 baseline
        gpu_train_cost = train_hours * gpu_train_rate

        print(f"Training Phase ({self.train_instance_type}):")
        print(f"  Time: {train_hours:.2f} hours")
        print(f"  Cost: ${train_cost:.2f}")
        print(f"  vs p5.48xlarge (H100) at same hours: ${gpu_train_cost:.2f}")

        # Inference costs (projected monthly, 24/7).
        monthly_hours = 24 * 30
        inferentia_monthly = inference_result["hourly_cost"] * monthly_hours
        gpu_infer_rate = ILLUSTRATIVE_HOURLY_USD["p3.2xlarge"]  # V100 serving baseline
        gpu_monthly = gpu_infer_rate * monthly_hours

        print("\nInference Phase (inf2.xlarge):")
        print(f"  Hourly: ${inference_result['hourly_cost']:.3f}")
        print(f"  Monthly (24/7): ${inferentia_monthly:.2f}")
        print(f"  vs p3.2xlarge (V100) monthly: ${gpu_monthly:.2f}")

        # Total (training + one month of serving) — savings computed, never hardcoded.
        total_aws_cost = train_cost + inferentia_monthly
        total_gpu_cost = gpu_train_cost + gpu_monthly
        savings = total_gpu_cost - total_aws_cost
        savings_pct = (savings / total_gpu_cost * 100) if total_gpu_cost else 0.0

        print("\nTotal (training + 1 month serving):")
        print(f"  AWS Neuron: ${total_aws_cost:.2f}")
        print(f"  GPU baseline: ${total_gpu_cost:.2f}")
        print(f"  💰 Illustrative savings: ${savings:.2f} ({savings_pct:.1f}%)")

        # Save report (all figures illustrative, derived from ILLUSTRATIVE_HOURLY_USD).
        report = {
            "project": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "Illustrative estimates from ILLUSTRATIVE_HOURLY_USD; verify AWS pricing.",
            "training": training_result,
            "inference": inference_result,
            "cost_comparison": {
                "monthly_inferentia": inferentia_monthly,
                "monthly_gpu_baseline": gpu_monthly,
                "monthly_savings": gpu_monthly - inferentia_monthly,
                "savings_percentage": (
                    (gpu_monthly - inferentia_monthly) / gpu_monthly * 100
                    if gpu_monthly
                    else 0.0
                ),
                "total_aws": total_aws_cost,
                "total_gpu_baseline": total_gpu_cost,
                "total_savings": savings,
            },
        }

        report_key = f"reports/{self.project_name}/cost_comparison.json"
        self.s3.put_object(
            Bucket=self.s3_bucket, Key=report_key, Body=json.dumps(report, indent=2)
        )

        print(f"\n✅ Full report saved to: s3://{self.s3_bucket}/{report_key}")

        return report


def run_pipeline(project_name: str, s3_bucket: str, config: dict[str, Any]) -> dict:
    """Run the full train→serve→compare pipeline. **Launches real, billable instances.**"""
    pipeline = TrainiumToInferentiaPipeline(project_name, s3_bucket)

    print("🚀 Starting train→serve pipeline...")
    training_result = pipeline.train_on_trainium(
        model_class=config.get("model_class", "AutoModelForSequenceClassification"),
        dataset_path=config["dataset_path"],
        config=config,
    )
    inference_result = pipeline.deploy_on_inferentia(
        model_path=training_result["model_path"],
        config={"inference_hours": config.get("inference_hours", 24)},
    )
    report = pipeline.run_cost_comparison(training_result, inference_result)
    print("\n🎉 Pipeline complete!")
    print(f"📊 Results: s3://{pipeline.s3_bucket}/experiments/{pipeline.project_name}/")
    return report


def main() -> None:
    """CLI entrypoint. Dry by default — it does NOT launch instances unless you pass --run.

    This mirrors the tutorial's other hardware-touching examples: running the file with no flags
    explains what *would* happen (and what it costs) rather than silently provisioning EC2.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--run", action="store_true", help="Actually launch instances (costs money)."
    )
    p.add_argument("--project", default="train-serve-demo")
    p.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket for scripts/data/results (required for --run).",
    )
    p.add_argument(
        "--dataset-path", default=None, help="s3://… tarball with your train.csv."
    )
    p.add_argument("--instance-type", default="trn1.32xlarge")
    args = p.parse_args()

    if not args.run:
        print(
            "DRY RUN — nothing launched.\n"
            "This example trains on Trainium, compiles the model once with torch_neuronx.trace()\n"
            "for inference, then serves it on an Inf2 instance behind an HTTP endpoint.\n\n"
            "It provisions REAL, billable EC2 instances and serves an UNAUTHENTICATED endpoint, so\n"
            "review the code and supply your own dataset first. Then re-run with, e.g.:\n"
            "    python trainium_to_inferentia_pipeline.py --run \\\n"
            "        --bucket my-ml-bucket --dataset-path s3://my-ml-bucket/data.tar.gz\n\n"
            "See VERSION_MATRIX.md: for new serving, prefer Trn2 + NxD Inference over Inf2."
        )
        return

    if not args.bucket or not args.dataset_path:
        p.error("--run requires --bucket and --dataset-path")

    run_pipeline(
        args.project,
        args.bucket,
        {
            "instance_type": args.instance_type,
            "dataset_path": args.dataset_path,
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "num_labels": 2,
            "inference_hours": 24 * 7,
        },
    )


if __name__ == "__main__":
    main()
