# examples/complete_workflow/trainium_to_inferentia_pipeline.py
import boto3
import torch
import torch_xla.core.xla_model as xm
import torch_neuronx
from datetime import datetime
import time
import json
import numpy as np

class TrainiumToInferentiaPipeline:
    """Complete pipeline from training to deployment with cost tracking"""
    
    def __init__(self, project_name, s3_bucket):
        self.project_name = project_name
        self.s3_bucket = s3_bucket
        self.ec2 = boto3.client('ec2')
        self.s3 = boto3.client('s3')
        
    def train_on_trainium(self, model_class, dataset_path, config):
        """Train model on Trainium with automatic cost tracking"""
        
        print("🚀 Phase 1: Training on Trainium")
        start_time = time.time()
        instance_type = config.get('instance_type', 'trn1.32xlarge')
        
        # Generate training script
        training_script = self._generate_training_script(model_class, dataset_path, config)
        
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
            Bucket=self.s3_bucket,
            Key='scripts/train.py',
            Body=training_script
        )
        
        # Launch instance
        response = self.ec2.run_instances(
            ImageId='ami-0abcdef1234567890',  # Deep Learning AMI
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            InstanceMarketOptions={
                'MarketType': 'spot',
                'SpotOptions': {
                    'SpotInstanceType': 'one-time',
                    'InstanceInterruptionBehavior': 'terminate'
                }
            },
            IamInstanceProfile={'Name': 'ML-Research-EC2-Role'},
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'Training-{self.project_name}'},
                    {'Key': 'Phase', 'Value': 'Training'},
                    {'Key': 'Project', 'Value': self.project_name}
                ]
            }]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        print(f"✅ Launched Trainium instance: {instance_id}")
        
        # Wait for training to complete
        training_time, training_cost = self._wait_for_training_completion(instance_id)
        
        return {
            'instance_id': instance_id,
            'training_time_hours': training_time,
            'training_cost_usd': training_cost,
            'model_path': f's3://{self.s3_bucket}/experiments/{self.project_name}/model_inferentia.pt'
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
echo "sudo shutdown -h now" | at now + {config.get('inference_hours', 24)} hours
"""
        
        # Upload inference script
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key='scripts/inference_server.py',
            Body=inference_script
        )
        
        # Launch Inferentia instance
        response = self.ec2.run_instances(
            ImageId='ami-0abcdef1234567890',  # Deep Learning AMI
            InstanceType='inf2.xlarge',
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            IamInstanceProfile={'Name': 'ML-Research-EC2-Role'},
            SecurityGroups=['ML-Research-SG'],
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'Inference-{self.project_name}'},
                    {'Key': 'Phase', 'Value': 'Inference'},
                    {'Key': 'Project', 'Value': self.project_name}
                ]
            }]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        
        # Wait for instance to be running and get public IP
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        
        print(f"✅ Deployed on Inferentia: {instance_id}")
        print(f"🌐 Inference endpoint: http://{public_ip}:8080/predict")
        print(f"💰 Cost: $0.227/hour (spot) - $0.758/hour (on-demand)")
        
        return {
            'instance_id': instance_id,
            'endpoint': f'http://{public_ip}:8080/predict',
            'hourly_cost': 0.227
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
    
    # Setup device
    device = xm.xla_device()
    print(f"Using device: {{device}}")
    
    # Load model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels={config.get('num_labels', 2)}
    )
    
    # Move to device
    model = model.to(device)
    
    # Compile for Neuron (training)
    print("🔧 Compiling model for Neuron...")
    example_input = {{
        'input_ids': torch.randint(0, 1000, (1, 512)),
        'attention_mask': torch.ones(1, 512)
    }}
    
    model = torch_neuronx.trace(
        model,
        (example_input['input_ids'].to(device), example_input['attention_mask'].to(device)),
        compiler_args=[
            '--model-type=transformer',
            '--enable-saturate-infinity',
            '--neuroncore-pipeline-cores=16'
        ]
    )
    print("✅ Model compilation complete")
    
    # Setup training
    dataset = ClimateDataset('dataset/train.csv')
    train_loader = DataLoader(dataset, batch_size={config.get('batch_size', 32)}, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr={config.get('learning_rate', 2e-5)})
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range({config.get('epochs', 10)}):
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
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {{epoch}}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {{epoch}} completed. Average Loss: {{avg_loss:.4f}}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Compile for Inferentia (inference)
            print("🔧 Compiling model for Inferentia...")
            inference_model = torch_neuronx.trace(
                model,
                (example_input['input_ids'].to(device)[:1], example_input['attention_mask'].to(device)[:1]),
                compiler_args=[
                    '--model-type=transformer',
                    '--static-weights',
                    '--batching_en',
                    '--max-batch-size=32'
                ]
            )
            
            # Save for Inferentia
            torch.jit.save(inference_model, '/home/ubuntu/results/model_inferentia.pt')
            
            # Save checkpoint
            torch.save({{
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }}, '/home/ubuntu/results/checkpoint.pt')
            
            print(f"✅ Saved new best model (loss: {{best_loss:.4f}})")
    
    print("🎉 Training completed!")
    
    # Generate training report
    report = {{
        'project': '{self.project_name}',
        'training_completed': datetime.now().isoformat(),
        'final_loss': best_loss,
        'epochs_trained': {config.get('epochs', 10)},
        'batch_size': {config.get('batch_size', 32)},
        'learning_rate': {config.get('learning_rate', 2e-5)}
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
        
        return f"""
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
    
    start_time = time.time()
    
    try:
        # Get input data
        data = request.json
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({{'error': 'No texts provided'}}), 400
        
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
        latency_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        request_count += len(texts)
        total_latency += latency_ms
        avg_latency = total_latency / request_count if request_count > 0 else 0
        
        # Calculate costs
        hourly_rate = 0.227  # inf2.xlarge spot price
        runtime_hours = (datetime.now() - start_time).total_seconds() / 3600
        total_cost = runtime_hours * hourly_rate
        cost_per_1k_requests = (total_cost / request_count) * 1000 if request_count > 0 else 0
        
        return jsonify({{
            'predictions': predictions.tolist(),
            'latency_ms': round(latency_ms, 2),
            'batch_size': len(texts),
            'statistics': {{
                'total_requests': request_count,
                'average_latency_ms': round(avg_latency, 2),
                'total_cost_usd': round(total_cost, 4),
                'cost_per_1k_requests': round(cost_per_1k_requests, 4),
                'requests_per_dollar': round(1000 / cost_per_1k_requests, 0) if cost_per_1k_requests > 0 else 0
            }}
        }})
        
    except Exception as e:
        return jsonify({{'error': str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{
        'status': 'healthy',
        'model': 'loaded' if model is not None else 'not_loaded',
        'uptime_hours': round((datetime.now() - start_time).total_seconds() / 3600, 2),
        'total_requests': request_count
    }})

@app.route('/stats', methods=['GET'])
def stats():
    runtime_hours = (datetime.now() - start_time).total_seconds() / 3600
    hourly_rate = 0.227
    total_cost = runtime_hours * hourly_rate
    
    return jsonify({{
        'runtime_hours': round(runtime_hours, 2),
        'total_requests': request_count,
        'total_cost_usd': round(total_cost, 4),
        'cost_per_request': round(total_cost / request_count, 6) if request_count > 0 else 0,
        'requests_per_hour': round(request_count / runtime_hours, 0) if runtime_hours > 0 else 0,
        'average_latency_ms': round(total_latency / request_count, 2) if request_count > 0 else 0
    }})

if __name__ == '__main__':
    load_model()
    print("🚀 Starting inference server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
"""
    
    def _get_cost_monitor_script(self, instance_type):
        """Generate cost monitoring script"""
        
        hourly_rates = {
            'trn1.2xlarge': 0.40,
            'trn1.32xlarge': 6.45,
            'trn2.48xlarge': 12.00
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
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            
            if state == 'terminated':
                end_time = time.time()
                training_time = (end_time - start_time) / 3600
                training_cost = training_time * 6.45  # Spot price for trn1.32xlarge
                
                print(f"✅ Training completed!")
                print(f"⏱️  Training time: {training_time:.2f} hours")
                print(f"💰 Training cost: ${training_cost:.2f}")
                
                return training_time, training_cost
            
            elif state == 'running':
                elapsed = (time.time() - start_time) / 3600
                estimated_cost = elapsed * 6.45
                print(f"🔄 Training in progress... {elapsed:.1f}h elapsed, ${estimated_cost:.2f} spent")
                
            time.sleep(300)  # Check every 5 minutes
    
    def run_cost_comparison(self, training_result, inference_result):
        """Generate cost comparison report"""
        
        print("\\n📊 Cost Comparison Report")
        print("=" * 50)
        
        # Training costs
        print(f"Training Phase (Trainium):")
        print(f"  Time: {training_result['training_time_hours']:.2f} hours")
        print(f"  Cost: ${training_result['training_cost_usd']:.2f}")
        print(f"  vs H100 (estimated): ${training_result['training_cost_usd'] * 2.5:.2f}")
        print(f"  Savings: ${training_result['training_cost_usd'] * 1.5:.2f} (60%)")
        
        # Inference costs (projected monthly)
        monthly_hours = 24 * 30
        inferentia_monthly = inference_result['hourly_cost'] * monthly_hours
        gpu_monthly = 3.06 * monthly_hours  # p3.2xlarge
        
        print(f"\\nInference Phase (Inferentia):")
        print(f"  Hourly: ${inference_result['hourly_cost']:.3f}")
        print(f"  Monthly (24/7): ${inferentia_monthly:.2f}")
        print(f"  vs GPU monthly: ${gpu_monthly:.2f}")
        print(f"  Monthly savings: ${gpu_monthly - inferentia_monthly:.2f} ({((gpu_monthly - inferentia_monthly) / gpu_monthly * 100):.1f}%)")
        
        # Total savings
        total_aws_cost = training_result['training_cost_usd'] + inferentia_monthly
        total_gpu_cost = (training_result['training_cost_usd'] * 2.5) + gpu_monthly
        
        print(f"\\nTotal Monthly Cost (Training + Inference):")
        print(f"  AWS ML Chips: ${total_aws_cost:.2f}")
        print(f"  Traditional GPU: ${total_gpu_cost:.2f}")
        print(f"  💰 Total Savings: ${total_gpu_cost - total_aws_cost:.2f}/month ({((total_gpu_cost - total_aws_cost) / total_gpu_cost * 100):.1f}%)")
        
        # Save report
        report = {
            'project': self.project_name,
            'timestamp': datetime.now().isoformat(),
            'training': training_result,
            'inference': inference_result,
            'cost_comparison': {
                'monthly_inferentia': inferentia_monthly,
                'monthly_gpu': gpu_monthly,
                'monthly_savings': gpu_monthly - inferentia_monthly,
                'savings_percentage': ((gpu_monthly - inferentia_monthly) / gpu_monthly * 100),
                'total_monthly_aws': total_aws_cost,
                'total_monthly_gpu': total_gpu_cost,
                'total_savings': total_gpu_cost - total_aws_cost
            }
        }
        
        report_key = f'reports/{self.project_name}/cost_comparison.json'
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2)
        )
        
        print(f"\\n✅ Full report saved to: s3://{self.s3_bucket}/{report_key}")
        
        return report

# Example usage script
def main():
    """Run complete Trainium to Inferentia pipeline"""
    
    # Configuration
    pipeline = TrainiumToInferentiaPipeline(
        project_name='climate-prediction-demo',
        s3_bucket='your-ml-experiments-bucket'  # Change this!
    )
    
    # Phase 1: Train on Trainium
    print("🚀 Starting complete ML pipeline demo...")
    
    training_config = {
        'instance_type': 'trn1.32xlarge',
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'num_labels': 2
    }
    
    training_result = pipeline.train_on_trainium(
        model_class='ClimateModel',
        dataset_path='datasets/climate_sentiment.tar.gz',
        config=training_config
    )
    
    # Phase 2: Deploy on Inferentia
    inference_config = {
        'inference_hours': 24 * 7  # 1 week
    }
    
    inference_result = pipeline.deploy_on_inferentia(
        model_path=training_result['model_path'],
        config=inference_config
    )
    
    # Phase 3: Cost analysis
    cost_report = pipeline.run_cost_comparison(training_result, inference_result)
    
    # Test the deployment
    print("\\n🧪 Testing inference endpoint...")
    import requests
    
    test_data = {
        'texts': [
            "Climate change is causing severe weather patterns",
            "Renewable energy adoption is accelerating globally",
            "Carbon emissions continue to rise despite commitments"
        ]
    }
    
    try:
        response = requests.post(inference_result['endpoint'], json=test_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference test successful!")
            print(f"   Predictions: {len(result['predictions'])} results")
            print(f"   Latency: {result['latency_ms']:.2f}ms")
            print(f"   Cost per 1k requests: ${result['statistics']['cost_per_1k_requests']:.4f}")
        else:
            print(f"❌ Inference test failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to inference endpoint: {e}")
        print("   The instance may still be starting up. Try again in a few minutes.")
    
    print("\\n🎉 Pipeline demo completed!")
    print(f"📊 Check full results at: s3://{pipeline.s3_bucket}/experiments/{pipeline.project_name}/")

if __name__ == "__main__":
    main()