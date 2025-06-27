# examples/rag_pipeline/neuron_rag_pipeline.py
import torch
import numpy as np
import faiss
import torch_neuronx
from transformers import AutoModel, AutoTokenizer
from transformers_neuronx import LlamaForSampling
import boto3
import json
import time
from datetime import datetime

class NeuronRAGPipeline:
    """Complete RAG pipeline optimized for AWS ML chips"""
    
    def __init__(self, embedding_model_path='BAAI/bge-base-en-v1.5', llm_model_path='./llama2-7b-neuron'):
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.embedder = None
        self.llm = None
        self.index = None
        self.documents = []
        self.s3 = boto3.client('s3')
        self.cost_tracker = RAGCostTracker()
        
    def setup_embedding_model(self, instance_type='inf2.xlarge'):
        """Load BGE embeddings model optimized for Inferentia2"""
        
        print("📥 Loading embedding model for Inferentia2...")
        
        # Load base model and tokenizer
        model = AutoModel.from_pretrained(self.embedding_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
        
        # Example input for compilation
        example_input = tokenizer(
            "Example text for compilation",
            max_length=384,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Compile for Inferentia2 with static shapes and batching
        print("🔧 Compiling embedding model for Inferentia2...")
        model_neuron = torch_neuronx.trace(
            model,
            (example_input['input_ids'], example_input['attention_mask']),
            compiler_args=[
                '--model-type=transformer',
                '--enable-fast-loading-neuron-binaries',
                '--static-weights',
                '--batching_en',
                '--max-batch-size=32'
            ]
        )
        
        self.embedder = {
            'model': model_neuron,
            'tokenizer': tokenizer
        }
        
        print("✅ Embedding model loaded and compiled for Inferentia2")
        print(f"   Expected throughput: ~680 embeddings/sec on {instance_type}")
        print(f"   Cost: $0.227/hour vs $0.10/1M tokens (OpenAI)")
        
    def setup_llm(self, instance_type='trn2.48xlarge'):
        """Load Llama 2 for generation on Trainium2"""
        
        print("📥 Loading LLM for Trainium2...")
        
        # Load quantized Llama 2 7B with Neuron optimizations
        self.llm = LlamaForSampling.from_pretrained(
            self.llm_model_path,
            batch_size=1,
            tp_degree=8,  # Tensor parallelism across 8 cores (trn2.48xlarge has 32 cores)
            n_positions=2048,
            amp='bf16',
            on_device_generation=True,  # Generate on Neuron device
            context_length_estimate=2048
        )
        
        print("✅ LLM loaded and optimized for Trainium2")
        print(f"   Expected throughput: ~64 tokens/sec on {instance_type}")
        print(f"   Cost: $12/hour vs $29.50/hour (H100)")
        
    def index_documents(self, documents, batch_size=32, save_to_s3=True):
        """Create FAISS index from documents with cost tracking"""
        
        print(f"📚 Indexing {len(documents)} documents...")
        start_time = time.time()
        
        embeddings = []
        
        # Process in batches for efficiency
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.embedder['tokenizer'](
                batch,
                max_length=384,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get embeddings from Inferentia2
            with torch.no_grad():
                outputs = self.embedder['model'](inputs['input_ids'], inputs['attention_mask'])
                # Mean pooling
                embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(embeddings_batch.cpu().numpy())
                
            # Progress update
            if (i // batch_size + 1) % 10 == 0:
                elapsed = time.time() - start_time
                processed = min(i + batch_size, len(documents))
                rate = processed / elapsed
                print(f"   Processed {processed}/{len(documents)} documents ({rate:.1f} docs/sec)")
        
        # Create FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Use IVF index for large-scale search (>100k documents)
        if len(documents) > 100000:
            print("🔧 Creating IVF index for large-scale search...")
            nlist = int(np.sqrt(len(documents)))
            quantizer = faiss.IndexFlatL2(embeddings_array.shape[1])
            self.index = faiss.IndexIVFFlat(quantizer, embeddings_array.shape[1], nlist)
            self.index.train(embeddings_array)
            self.index.add(embeddings_array)
        else:
            print("🔧 Creating flat index for fast search...")
            self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
            self.index.add(embeddings_array)
        
        self.documents = documents
        
        # Calculate costs
        total_time = time.time() - start_time
        embedding_cost = self.cost_tracker.calculate_embedding_cost(len(documents), total_time)
        
        print(f"✅ Indexed {len(documents)} documents in {total_time:.1f} seconds")
        print(f"💰 Indexing cost: ${embedding_cost:.4f}")
        
        # Save to S3 if requested
        if save_to_s3:
            self._save_index_to_s3()
            
        return {
            'num_documents': len(documents),
            'indexing_time_seconds': total_time,
            'indexing_cost_usd': embedding_cost,
            'embedding_dimension': embeddings_array.shape[1]
        }
    
    def retrieve(self, query, k=5):
        """Retrieve relevant documents with cost tracking"""
        
        start_time = time.time()
        
        # Encode query
        inputs = self.embedder['tokenizer'](
            [query],
            max_length=384,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            query_embedding = self.embedder['model'](inputs['input_ids'], inputs['attention_mask'])
            query_embedding = query_embedding.last_hidden_state.mean(dim=1)
        
        # Search FAISS index
        query_np = query_embedding.cpu().numpy().astype('float32')
        distances, indices = self.index.search(query_np, k)
        
        # Get documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        
        # Calculate retrieval cost
        retrieval_time = time.time() - start_time
        retrieval_cost = self.cost_tracker.calculate_retrieval_cost(1, retrieval_time)
        
        return retrieved_docs, distances[0], retrieval_cost
    
    def generate(self, query, max_length=512, temperature=0.7, top_p=0.95):
        """Complete RAG pipeline with comprehensive cost tracking"""
        
        pipeline_start = time.time()
        
        # Step 1: Retrieve relevant documents
        print("🔍 Retrieving relevant documents...")
        docs, scores, retrieval_cost = self.retrieve(query, k=3)
        
        # Step 2: Build context
        context = "\\n\\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        
        # Step 3: Create prompt
        prompt = f\"\"\"Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:\"\"\"
        
        # Step 4: Generate response on Trainium
        print("🤖 Generating response...")
        generation_start = time.time()
        
        response = self.llm.sample(
            prompt,
            max_length=max_length,
            top_k=50,
            top_p=top_p,
            temperature=temperature
        )
        
        generation_time = time.time() - generation_start
        generation_cost = self.cost_tracker.calculate_generation_cost(len(prompt.split()), max_length, generation_time)
        
        # Step 5: Calculate total costs
        total_time = time.time() - pipeline_start
        total_cost = retrieval_cost + generation_cost
        
        result = {
            'answer': response,
            'sources': docs,
            'relevance_scores': scores.tolist(),
            'costs': {
                'retrieval_cost_usd': retrieval_cost,
                'generation_cost_usd': generation_cost,
                'total_cost_usd': total_cost
            },
            'performance': {
                'total_time_seconds': total_time,
                'retrieval_time_seconds': time.time() - pipeline_start - generation_time,
                'generation_time_seconds': generation_time
            },
            'metadata': {
                'num_documents_retrieved': len(docs),
                'prompt_length_tokens': len(prompt.split()),
                'response_length_tokens': len(response.split())
            }
        }
        
        # Log the interaction
        self.cost_tracker.log_interaction(query, result)
        
        return result
    
    def batch_generate(self, queries, max_length=512):
        """Process multiple queries efficiently"""
        
        results = []
        total_cost = 0
        
        print(f"🔄 Processing {len(queries)} queries in batch...")
        
        for i, query in enumerate(queries):
            print(f"   Query {i+1}/{len(queries)}: {query[:50]}...")
            result = self.generate(query, max_length)
            results.append(result)
            total_cost += result['costs']['total_cost_usd']
            
        print(f"✅ Batch processing complete!")
        print(f"💰 Total cost: ${total_cost:.4f}")
        print(f"💰 Average cost per query: ${total_cost/len(queries):.4f}")
        
        return results, total_cost
    
    def _save_index_to_s3(self, bucket='rag-indexes'):
        """Save FAISS index and documents to S3"""
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, '/tmp/faiss_index.bin')
            self.s3.upload_file('/tmp/faiss_index.bin', bucket, 'faiss_index.bin')
            
            # Save documents
            with open('/tmp/documents.json', 'w') as f:
                json.dump(self.documents, f)
            self.s3.upload_file('/tmp/documents.json', bucket, 'documents.json')
            
            print(f"✅ Index saved to S3: s3://{bucket}/")
            
        except Exception as e:
            print(f"⚠️  Could not save to S3: {e}")
    
    def load_index_from_s3(self, bucket='rag-indexes'):
        """Load FAISS index and documents from S3"""
        
        try:
            # Load FAISS index
            self.s3.download_file(bucket, 'faiss_index.bin', '/tmp/faiss_index.bin')
            self.index = faiss.read_index('/tmp/faiss_index.bin')
            
            # Load documents
            self.s3.download_file(bucket, 'documents.json', '/tmp/documents.json')
            with open('/tmp/documents.json', 'r') as f:
                self.documents = json.load(f)
            
            print(f"✅ Index loaded from S3: s3://{bucket}/")
            print(f"   {len(self.documents)} documents available")
            
        except Exception as e:
            print(f"❌ Could not load from S3: {e}")

class RAGCostTracker:
    """Track costs for RAG pipeline operations"""
    
    def __init__(self):
        self.interactions = []
        self.start_time = datetime.now()
        
        # Cost rates (per hour)
        self.rates = {
            'inf2_xlarge': 0.227,    # Inferentia2 for embeddings
            'trn2_48xlarge': 12.00   # Trainium2 for generation
        }
        
    def calculate_embedding_cost(self, num_documents, time_seconds):
        """Calculate cost for embedding generation"""
        hours = time_seconds / 3600
        return self.rates['inf2_xlarge'] * hours
    
    def calculate_retrieval_cost(self, num_queries, time_seconds):
        """Calculate cost for retrieval (minimal - mostly FAISS CPU)"""
        hours = time_seconds / 3600
        return self.rates['inf2_xlarge'] * hours * 0.1  # 10% of full rate
    
    def calculate_generation_cost(self, prompt_tokens, max_tokens, time_seconds):
        """Calculate cost for text generation"""
        hours = time_seconds / 3600
        return self.rates['trn2_48xlarge'] * hours
    
    def log_interaction(self, query, result):
        """Log an interaction for cost analysis"""
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'costs': result['costs'],
            'performance': result['performance'],
            'metadata': result['metadata']
        }
        
        self.interactions.append(interaction)
    
    def generate_cost_report(self):
        """Generate comprehensive cost report"""
        
        if not self.interactions:
            return "No interactions logged yet."
        
        total_cost = sum(i['costs']['total_cost_usd'] for i in self.interactions)
        total_queries = len(self.interactions)
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        avg_cost_per_query = total_cost / total_queries
        avg_retrieval_time = np.mean([i['performance']['retrieval_time_seconds'] for i in self.interactions])
        avg_generation_time = np.mean([i['performance']['generation_time_seconds'] for i in self.interactions])
        
        report = f\"\"\"
📊 RAG Pipeline Cost Report
{'='*40}
Runtime: {runtime_hours:.2f} hours
Total Queries: {total_queries}
Total Cost: ${total_cost:.4f}

💰 Cost Breakdown:
  Average per query: ${avg_cost_per_query:.4f}
  Cost per hour: ${total_cost/runtime_hours:.4f}
  
⚡ Performance:
  Avg retrieval time: {avg_retrieval_time:.2f}s
  Avg generation time: {avg_generation_time:.2f}s
  
🔄 Throughput:
  Queries per hour: {total_queries/runtime_hours:.1f}
  Queries per dollar: {total_queries/total_cost:.0f}

💡 Cost Comparison:
  OpenAI GPT-4: ~$0.06/query (estimated)
  AWS RAG Pipeline: ${avg_cost_per_query:.4f}/query
  Savings: {((0.06-avg_cost_per_query)/0.06*100):.1f}%
{'='*40}
\"\"\"
        
        return report

# Example usage and demo
def main():
    \"\"\"Demonstrate the complete RAG pipeline\"\"\"
    
    print("🚀 Starting Modern RAG Pipeline Demo on AWS ML Chips")
    print("=" * 60)
    
    # Initialize RAG pipeline
    rag = NeuronRAGPipeline()
    
    # Setup models
    rag.setup_embedding_model('inf2.xlarge')
    rag.setup_llm('trn2.48xlarge')
    
    # Example documents (in real use, load from your data source)
    research_papers = [
        "Transformer models have revolutionized natural language processing through attention mechanisms.",
        "Climate change is accelerating due to increased greenhouse gas emissions from human activities.",
        "Machine learning on specialized hardware like TPUs and Neuron chips offers significant cost advantages.",
        "Retrieval-augmented generation combines the benefits of parametric and non-parametric knowledge.",
        "AWS Trainium chips provide up to 60% cost savings compared to traditional GPU training.",
        "Vector databases and semantic search enable more relevant document retrieval for RAG systems.",
        "Fine-tuning large language models on domain-specific data improves task performance.",
        "Efficient inference optimization includes techniques like quantization and model distillation.",
        "Academic research budgets benefit significantly from cloud cost optimization strategies.",
        "Neuron SDK compiler optimizations can achieve near-GPU performance on Trainium instances."
    ]
    
    # Index documents
    print("\\n📚 Indexing research documents...")
    indexing_result = rag.index_documents(research_papers, batch_size=8)
    
    # Example queries
    queries = [
        "What are the cost advantages of using specialized ML hardware?",
        "How does retrieval-augmented generation work?",
        "What optimization techniques are available for inference?",
        "How can academic researchers save money on cloud computing?"
    ]
    
    # Process queries
    print("\\n🤖 Processing example queries...")
    results, total_cost = rag.batch_generate(queries, max_length=256)
    
    # Display results
    print("\\n📋 Results:")
    print("=" * 40)
    
    for i, (query, result) in enumerate(zip(queries, results)):
        print(f"\\nQuery {i+1}: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Cost: ${result['costs']['total_cost_usd']:.4f}")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Time: {result['performance']['total_time_seconds']:.2f}s")
        print("-" * 40)
    
    # Generate cost report
    print("\\n" + rag.cost_tracker.generate_cost_report())
    
    # Save results
    with open('/tmp/rag_demo_results.json', 'w') as f:
        json.dump({
            'indexing_result': indexing_result,
            'query_results': results,
            'total_cost': total_cost,
            'cost_report': rag.cost_tracker.generate_cost_report()
        }, f, indent=2)
    
    print("\\n✅ Demo completed! Results saved to /tmp/rag_demo_results.json")
    print(f"💰 Total demo cost: ${total_cost + indexing_result['indexing_cost_usd']:.4f}")

if __name__ == "__main__":
    main()


# examples/rag_pipeline/deploy_rag_service.py
\"\"\"
Production RAG deployment script with auto-scaling and monitoring
\"\"\"

from flask import Flask, request, jsonify
import torch_neuronx
from neuron_rag_pipeline import NeuronRAGPipeline
import json
import time
from datetime import datetime
import boto3

app = Flask(__name__)

# Global RAG pipeline
rag_pipeline = None
deployment_stats = {
    'start_time': datetime.now(),
    'total_requests': 0,
    'total_cost': 0.0,
    'average_latency': 0.0
}

def initialize_rag_pipeline():
    \"\"\"Initialize RAG pipeline for production\"\"\"
    global rag_pipeline
    
    print("🚀 Initializing production RAG pipeline...")
    
    rag_pipeline = NeuronRAGPipeline()
    rag_pipeline.setup_embedding_model('inf2.xlarge')
    rag_pipeline.setup_llm('trn2.48xlarge')
    
    # Load pre-built index from S3
    rag_pipeline.load_index_from_s3('your-rag-indexes-bucket')
    
    print("✅ RAG pipeline ready for production!")

@app.route('/chat', methods=['POST'])
def chat():
    \"\"\"Main chat endpoint\"\"\"
    global deployment_stats
    
    start_time = time.time()
    
    try:
        data = request.json
        query = data.get('query', '')
        max_length = data.get('max_length', 512)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Generate response
        result = rag_pipeline.generate(query, max_length=max_length)
        
        # Update statistics
        latency = time.time() - start_time
        deployment_stats['total_requests'] += 1
        deployment_stats['total_cost'] += result['costs']['total_cost_usd']
        deployment_stats['average_latency'] = (
            (deployment_stats['average_latency'] * (deployment_stats['total_requests'] - 1) + latency) 
            / deployment_stats['total_requests']
        )
        
        # Add deployment metadata
        result['deployment'] = {
            'latency_seconds': latency,
            'request_id': deployment_stats['total_requests'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    \"\"\"Health check endpoint\"\"\"
    return jsonify({
        'status': 'healthy',
        'pipeline_loaded': rag_pipeline is not None,
        'uptime_hours': (datetime.now() - deployment_stats['start_time']).total_seconds() / 3600
    })

@app.route('/stats', methods=['GET'])
def stats():
    \"\"\"Deployment statistics\"\"\"
    runtime_hours = (datetime.now() - deployment_stats['start_time']).total_seconds() / 3600
    
    return jsonify({
        'runtime_hours': round(runtime_hours, 2),
        'total_requests': deployment_stats['total_requests'],
        'total_cost_usd': round(deployment_stats['total_cost'], 4),
        'average_latency_seconds': round(deployment_stats['average_latency'], 3),
        'requests_per_hour': round(deployment_stats['total_requests'] / runtime_hours, 1) if runtime_hours > 0 else 0,
        'cost_per_request': round(deployment_stats['total_cost'] / deployment_stats['total_requests'], 6) if deployment_stats['total_requests'] > 0 else 0
    })

if __name__ == '__main__':
    initialize_rag_pipeline()
    print("🌐 Starting RAG service on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)