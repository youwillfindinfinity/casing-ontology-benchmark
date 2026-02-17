"""
Embedding Model Evaluation Pipeline - No API Keys Required
Evaluates different embedding models using your FAISS indices
"""

import pandas as pd
import numpy as np
import faiss
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import pickle
import warnings
warnings.filterwarnings("ignore")

class EmbeddingModelEvaluator:
    def __init__(self, base_dir="INPUT", output_dir="OUTPUTS"):
        self.base_dir = base_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Map your indices to their corresponding models
        self.embedding_models = {
            'allmpnet_bioasq': {
                'model_name': 'juanpablomesa/all-mpnet-base-v2-bioasq-matryoshka',
                'index_file': 'ncbi_faiss_allmpnetbasev2bioasqmatryoshka.index',
                'description': 'All-MPNet BioBASQ Matryoshka'
            },
            'bge_base': {
                'model_name': 'BAAI/bge-base-en-v1.5',
                'index_file': 'ncbi_faiss_bgebaseenv15.index',
                'description': 'BGE Base English v1.5'
            },
            'bios_minilm': {
                'model_name': 'menadsa/BioS-MiniLM',
                'index_file': 'ncbi_faiss_biosminilm.index',
                'description': 'BioS MiniLM'
            },
            'e5_large': {
                'model_name': 'intfloat/e5-large-v2',
                'index_file': 'ncbi_faiss_e5largev2.index',
                'description': 'E5 Large v2'
            },
            'e5_small': {
                'model_name': 'intfloat/e5-small-v2',
                'index_file': 'ncbi_faiss_e5smallv2.index',
                'description': 'E5 Small v2'
            },
            'multilingual_e5': {
                'model_name': 'intfloat/multilingual-e5-large',
                'index_file': 'ncbi_faiss_multilinguale5large.index',
                'description': 'Multilingual E5 Large'
            },
            'pubmedbert': {
                'model_name': 'NeuML/pubmedbert-base-embeddings',
                'index_file': 'ncbi_faiss_pubmedbertbaseembeddings.index',
                'description': 'PubMedBERT Embeddings'
            },
            'sbiobert': {
                'model_name': 'pritamdeka/S-BioBert-snli-multinli-stsb',
                'index_file': 'ncbi_faiss_sbiobertsnlimultinlistsb.index',
                'description': 'S-BioBERT SNLI MultiNLI STSB'
            }
        }
        
        # Load taxon data for evaluation
        self.load_taxon_data()
        
    def load_taxon_data(self):
        """Load taxonomic data for evaluation"""
        print("🔄 Loading taxonomic data...")
        
        try:
            with open(f"{self.base_dir}/taxon_names.pkl", "rb") as f:
                self.taxon_names = pickle.load(f)
            print(f"✅ Loaded {len(self.taxon_names)} taxon names")
        except FileNotFoundError:
            print("❌ Could not load taxon_names.pkl")
            self.taxon_names = []
        
        try:
            with open(f"{self.base_dir}/taxon_data_r.pkl", "rb") as f:
                self.taxon_data = pickle.load(f)
            print(f"✅ Loaded {len(self.taxon_data)} taxon data entries")
        except FileNotFoundError:
            print("❌ Could not load taxon_data_r.pkl")
            self.taxon_data = []

    def evaluate_embedding_model(self, model_key, model_info):
        """Evaluate a single embedding model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_info['description']}")
        print(f"Model: {model_info['model_name']}")
        print(f"Index: {model_info['index_file']}")
        print(f"{'='*60}")
        
        results = {
            'model': model_key,
            'description': model_info['description'],
            'model_name': model_info['model_name']
        }
        
        # Check if index file exists
        index_path = f"{self.base_dir}/indices/{model_info['index_file']}"
        if not os.path.exists(index_path):
            print(f"❌ Index file not found: {index_path}")
            return None
        
        try:
            # Load the FAISS index
            index = faiss.read_index(index_path)
            results['index_size'] = index.ntotal
            results['dimension'] = index.d
            
            print(f"📊 Index loaded: {index.ntotal:,} vectors, dimension {index.d}")
            
            # File size
            file_size = os.path.getsize(index_path) / (1024**3)  # GB
            results['file_size_gb'] = file_size
            print(f"💾 File size: {file_size:.2f} GB")
            
            # Load the actual embedding model for testing
            print("🔄 Loading sentence transformer model...")
            model = SentenceTransformer(model_info['model_name'])
            
            # Test queries related to your research domain
            test_queries = [
                "burn injury treatment",
                "skin tissue damage",
                "wound healing process", 
                "inflammatory response",
                "cytokine signaling",
                "complement activation",
                "neutrophil infiltration",
                "tissue repair mechanisms",
                "biological taxonomy classification",
                "NCBI taxonomic data"
            ]
            
            # Evaluate retrieval performance
            retrieval_results = self.test_retrieval_performance(model, index, test_queries)
            results.update(retrieval_results)
            
            print(f"✅ Evaluation completed for {model_key}")
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_key}: {e}")
            return None

    def test_retrieval_performance(self, model, index, test_queries):
        """Test retrieval performance with biological queries"""
        print("🔄 Testing retrieval performance...")
        
        results = {}
        all_similarities = []
        
        for i, query in enumerate(test_queries):
            try:
                # Encode query
                query_embedding = model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
                
                # Search index
                k = min(10, index.ntotal)  # Top-k results
                similarities, indices = index.search(query_embedding, k)
                
                # Store similarity scores
                avg_similarity = similarities[0].mean()
                max_similarity = similarities[0].max()
                min_similarity = similarities[0].min()
                
                all_similarities.extend(similarities[0])
                
                print(f"  Query {i+1}: avg_sim={avg_similarity:.3f}, max_sim={max_similarity:.3f}")
                
            except Exception as e:
                print(f"  Query {i+1} failed: {e}")
                continue
        
        if all_similarities:
            results['avg_similarity'] = np.mean(all_similarities)
            results['std_similarity'] = np.std(all_similarities)
            results['max_similarity'] = np.max(all_similarities)
            results['min_similarity'] = np.min(all_similarities)
            results['queries_tested'] = len(test_queries)
        
        return results

    def run_comprehensive_evaluation(self):
        """Run evaluation on all embedding models"""
        print("🚀 STARTING COMPREHENSIVE EMBEDDING EVALUATION")
        print("="*70)
        
        all_results = []
        
        for model_key, model_info in self.embedding_models.items():
            result = self.evaluate_embedding_model(model_key, model_info)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("❌ No successful evaluations")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.output_dir}/embedding_evaluation_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Generate summary and visualizations
        self.create_evaluation_summary(results_df, timestamp)
        self.create_visualizations(results_df, timestamp)
        
        return results_df

    def create_evaluation_summary(self, results_df, timestamp):
        """Create evaluation summary"""
        print("\n📊 EVALUATION SUMMARY")
        print("="*50)
        
        # Performance ranking
        if 'avg_similarity' in results_df.columns:
            ranked = results_df.sort_values('avg_similarity', ascending=False)
            
            print("\n🏆 RANKING BY AVERAGE SIMILARITY:")
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                print(f"{i:2d}. {row['description']:30s} {row['avg_similarity']:.3f}")
        
        # Size comparison
        if 'file_size_gb' in results_df.columns:
            print(f"\n💾 FILE SIZES:")
            size_sorted = results_df.sort_values('file_size_gb')
            for _, row in size_sorted.iterrows():
                print(f"  {row['description']:30s} {row['file_size_gb']:6.2f} GB")
        
        # Dimension comparison
        if 'dimension' in results_df.columns:
            print(f"\n📐 DIMENSIONS:")
            dim_counts = results_df['dimension'].value_counts().sort_index()
            for dim, count in dim_counts.items():
                models = results_df[results_df['dimension']==dim]['description'].tolist()
                print(f"  {dim:4d}D: {count} models - {', '.join(models)}")
        
        # Save summary
        summary_file = f"{self.output_dir}/embedding_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("EMBEDDING MODEL EVALUATION SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Models evaluated: {len(results_df)}\n\n")
            
            if 'avg_similarity' in results_df.columns:
                f.write("PERFORMANCE RANKING:\n")
                ranked = results_df.sort_values('avg_similarity', ascending=False)
                for i, (_, row) in enumerate(ranked.iterrows(), 1):
                    f.write(f"{i:2d}. {row['description']:30s} {row['avg_similarity']:.3f}\n")
        
        print(f"📄 Summary saved to: {summary_file}")

    def create_visualizations(self, results_df, timestamp):
        """Create visualization plots"""
        print("🎨 Creating visualizations...")
        
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Embedding Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Performance comparison
        if 'avg_similarity' in results_df.columns:
            ax1 = axes[0, 0]
            sorted_df = results_df.sort_values('avg_similarity')
            bars = ax1.barh(range(len(sorted_df)), sorted_df['avg_similarity'])
            ax1.set_yticks(range(len(sorted_df)))
            ax1.set_yticklabels([desc[:20] + '...' if len(desc) > 20 else desc 
                                for desc in sorted_df['description']], fontsize=9)
            ax1.set_xlabel('Average Similarity Score')
            ax1.set_title('Model Performance Comparison')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, sorted_df['avg_similarity'])):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', ha='left', va='center', fontsize=8)
        
        # 2. File size comparison
        if 'file_size_gb' in results_df.columns:
            ax2 = axes[0, 1]
            ax2.scatter(results_df['file_size_gb'], results_df['avg_similarity'] if 'avg_similarity' in results_df.columns else [1]*len(results_df))
            ax2.set_xlabel('File Size (GB)')
            ax2.set_ylabel('Average Similarity' if 'avg_similarity' in results_df.columns else 'Model Count')
            ax2.set_title('Size vs Performance Trade-off')
            
            # Add model labels
            for _, row in results_df.iterrows():
                ax2.annotate(row['description'][:15], 
                           (row['file_size_gb'], row['avg_similarity'] if 'avg_similarity' in results_df.columns else 1),
                           xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # 3. Dimension distribution
        if 'dimension' in results_df.columns:
            ax3 = axes[1, 0]
            dim_counts = results_df['dimension'].value_counts()
            ax3.pie(dim_counts.values, labels=[f'{dim}D' for dim in dim_counts.index], autopct='%1.0f%%')
            ax3.set_title('Dimension Distribution')
        
        # 4. Model summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for _, row in results_df.iterrows():
            table_data.append([
                row['description'][:20],
                f"{row['dimension'] if 'dimension' in row else 'N/A'}D",
                f"{row['file_size_gb']:.1f}GB" if 'file_size_gb' in row else 'N/A',
                f"{row['avg_similarity']:.3f}" if 'avg_similarity' in row else 'N/A'
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'Dim', 'Size', 'Perf'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax4.set_title('Model Summary Table')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"{self.output_dir}/embedding_evaluation_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()

def main():
    """Main execution function"""
    print("🔬 EMBEDDING MODEL EVALUATION PIPELINE")
    print("No API keys required - Direct embedding evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = EmbeddingModelEvaluator(base_dir="INPUT", output_dir="OUTPUTS")
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if results is not None:
        print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"📁 Check the OUTPUTS directory for:")
        print(f"   - Detailed CSV results")
        print(f"   - Summary text file") 
        print(f"   - Visualization plots")
        print(f"\n📊 Best performing model: {results.loc[results['avg_similarity'].idxmax()]['description'] if 'avg_similarity' in results.columns else 'N/A'}")
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()
