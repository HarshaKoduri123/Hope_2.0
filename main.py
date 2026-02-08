import json
import os
from typing import List, Dict, Any
from chunking_methods import ChunkingMethods
from hope_core import HOPEMetric
from rag_evaluation import RAGEvaluator
import numpy as np
import pandas as pd
from scipy import stats
from read_doc import read_documents
import matplotlib.pyplot as plt
import seaborn as sns
from rl.train_ppo import train_chunk_optimizer 
from rl.inference import optimize_chunks


class HOPEExperiment:
    def __init__(self):
        self.chunker = ChunkingMethods()
        self.hope_metric = HOPEMetric()
        self.rag_evaluator = RAGEvaluator()
    
    def load_documents(self, data_dir: str) -> List[Dict[str, Any]]:
        documents = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'id': filename.replace('.txt', ''),
                        'content': content,
                        'source': filename
                    })
        return documents
    
    def generate_questions(self, document: str, num_questions: int = 5) -> List[str]:
        prompt = f"""Generate {num_questions} questions that can be answered from the following text:
        
        {document[:1000]}...
        
        Questions:
        1."""
        
        response = self.hope_metric.query_llm(prompt)
        questions = self.hope_metric._parse_llm_output(response, num_questions)
        return questions
    
    def run_experiment(self, documents: List[Dict[str, Any]], output_file: str):
        results = []

        chunking_strategies = [
            ('fixed_small', lambda text: self.chunker.fixed_size_chunking(text, 500, 50)),
            ('fixed_large', lambda text: self.chunker.fixed_size_chunking(text, 2000, 200)),
            ('recursive_small', lambda text: self.chunker.recursive_chunking(text, 500)),
            ('recursive_large', lambda text: self.chunker.recursive_chunking(text, 2000)),
            ('semantic', lambda text: self.chunker.semantic_chunking(text)),
        ]

        for doc in documents:
            print(f"Processing document: {doc['id']}")
            questions = self.generate_questions(doc['content'], 5)
            ground_truths = questions

            for strategy_name, chunking_func in chunking_strategies:
                print(f"Testing strategy: {strategy_name}")

                try:
                    passages = chunking_func(doc['content'])
                    print(f"Generated {len(passages)} passages")
                except Exception as e:
                    print(f"Chunking failed: {e}")
                    continue

                hope_results = self.hope_metric.calculate_hope(
                    doc['content'],
                    passages,
                    images=doc.get("images", [])
                )

                rag_results = self.rag_evaluator.evaluate_rag(
                    questions, ground_truths, passages, [doc['content']]
                )
    

                result = {
                    'document_id': doc['id'],
                    'chunking_strategy': strategy_name,
                    'num_passages': len(passages),
                    'hope_metrics': hope_results,
                    'rag_metrics': rag_results
                }
                results.append(result)

                print(f"HOPE Score: {hope_results['hope_score']:.3f}")
                print(f"RAG Accuracy: {rag_results['answer_correctness']:.3f}")

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)


                # RL-BASED OPTIMIZATION 
                # print(" Training RL chunk optimizer (PPO)...") 
                # base_passages = self.chunker.semantic_chunking(doc['content']) 
                # model = train_chunk_optimizer( document=doc['content'], 
                # passages=base_passages,
                # hope_metric=self.hope_metric() 
                # optimized_passages = optimize_chunks( model, doc['content'], base_passages, self.hope_metric )
                # hope_results = self.hope_metric.calculate_hope( doc['content'], optimized_passages, images=doc.get("images", []) ) 
                # rag_results = self.rag_evaluator.evaluate_rag( questions, ground_truths, optimized_passages, [doc['content']] )
                # results.append({ 'document_id': doc['id'], 'chunking_strategy': 'rl_optimized', 'num_passages': len(optimized_passages), 'hope_metrics': hope_results, 'rag_metrics': rag_results })

        return results
    
    def analyze_results(self, results: List[Dict[str, Any]], analysis_file: str = "results/analysis.json"):

        data = []
        for r in results:
            data.append({
                'document_id': r['document_id'],
                'chunking_strategy': r['chunking_strategy'],
                'num_passages': r['num_passages'],
                'hope_score': r['hope_metrics']['hope_score'],
                'zeta_con': r['hope_metrics']['zeta_con'],
                'zeta_sem': r['hope_metrics']['zeta_sem'],
                'zeta_inf': r['hope_metrics']['zeta_inf'],
                'zeta_align': r['hope_metrics']['zeta_align'],
                'answer_correctness': r['rag_metrics']['answer_correctness'],
                'response_relevancy': r['rag_metrics']['response_relevancy'],
                'factual_correctness': r['rag_metrics']['factual_correctness'],
                'context_recall': r['rag_metrics']['context_recall']
            })
        
        df = pd.DataFrame(data)
        
        analysis = {
            'overall_summary': self._calculate_overall_summary(df),
            'by_strategy': self._calculate_by_strategy(df),
            'by_document': self._calculate_by_document(df),
            'correlations': self._calculate_correlations(df),
            'statistical_tests': self._perform_statistical_tests(df),
            'best_strategies': self._identify_best_strategies(df)
        }
        
        self._print_analysis_summary(analysis)
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            analysis_serializable = json.loads(json.dumps(analysis, cls=NumpyEncoder))
            json.dump(analysis_serializable, f, indent=2)
        self._generate_visualizations(df, analysis_file.replace('.json', '_plots'))
        
        return analysis
    
    def _calculate_overall_summary(self, df: pd.DataFrame) -> Dict[str, Any]:

        return {
            'total_experiments': len(df),
            'avg_hope_score': float(df['hope_score'].mean()),
            'std_hope_score': float(df['hope_score'].std()),
            'avg_rag_accuracy': float(df['answer_correctness'].mean()),
            'std_rag_accuracy': float(df['answer_correctness'].std()),
            'avg_passages_per_doc': float(df.groupby('document_id')['num_passages'].mean().mean()),
            'hope_components': {
                'zeta_con_mean': float(df['zeta_con'].mean()),
                'zeta_sem_mean': float(df['zeta_sem'].mean()),
                'zeta_inf_mean': float(df['zeta_inf'].mean()),
                'zeta_align_mean': float(df['zeta_align'].mean())
            },
            'rag_components': {
                'answer_correctness_mean': float(df['answer_correctness'].mean()),
                'response_relevancy_mean': float(df['response_relevancy'].mean()),
                'factual_correctness_mean': float(df['factual_correctness'].mean()),
                'context_recall_mean': float(df['context_recall'].mean())
            }
        }
    
    def _calculate_by_strategy(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by chunking strategy"""
        by_strategy = {}
        for strategy in df['chunking_strategy'].unique():
            strategy_df = df[df['chunking_strategy'] == strategy]
            by_strategy[strategy] = {
                'count': int(len(strategy_df)),
                'avg_hope_score': float(strategy_df['hope_score'].mean()),
                'avg_rag_accuracy': float(strategy_df['answer_correctness'].mean()),
                'avg_passages': float(strategy_df['num_passages'].mean()),
                'hope_components': {
                    'zeta_con': float(strategy_df['zeta_con'].mean()),
                    'zeta_sem': float(strategy_df['zeta_sem'].mean()),
                    'zeta_inf': float(strategy_df['zeta_inf'].mean()),
                    'zeta_align': float(strategy_df['zeta_align'].mean())
                },
                'rag_components': {
                    'answer_correctness': float(strategy_df['answer_correctness'].mean()),
                    'response_relevancy': float(strategy_df['response_relevancy'].mean()),
                    'factual_correctness': float(strategy_df['factual_correctness'].mean()),
                    'context_recall': float(strategy_df['context_recall'].mean())
                }
            }
        return by_strategy
    
    def _calculate_by_document(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:

        by_document = {}
        for doc_id in df['document_id'].unique():
            doc_df = df[df['document_id'] == doc_id]
            by_document[doc_id] = {
                'experiments': int(len(doc_df)),
                'avg_hope_score': float(doc_df['hope_score'].mean()),
                'avg_rag_accuracy': float(doc_df['answer_correctness'].mean()),
                'strategies_tested': doc_df['chunking_strategy'].unique().tolist()
            }
        return by_document
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:

        hope_metrics = ['hope_score', 'zeta_con', 'zeta_sem', 'zeta_inf', 'zeta_align']
        rag_metrics = ['answer_correctness', 'response_relevancy', 'factual_correctness', 'context_recall']
        
        correlations = {}
        
        for hope_metric in hope_metrics:
            correlations[hope_metric] = {}
            for rag_metric in rag_metrics:
                pearson_r, pearson_p = stats.pearsonr(df[hope_metric], df[rag_metric])
                spearman_rho, spearman_p = stats.spearmanr(df[hope_metric], df[rag_metric])
                
                correlations[hope_metric][rag_metric] = {
                    'pearson': {
                        'correlation': float(pearson_r),
                        'p_value': float(pearson_p),
                        'significant': pearson_p < 0.05
                    },
                    'spearman': {
                        'correlation': float(spearman_rho),
                        'p_value': float(spearman_p),
                        'significant': spearman_p < 0.05
                    }
                }
        
        return correlations
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:

        tests = {}
        strategy_groups = [group['hope_score'].values for name, group in df.groupby('chunking_strategy')]
        if len(strategy_groups) > 1:
            f_stat, p_value = stats.f_oneway(*strategy_groups)
            tests['anova_hope_scores'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Tests if HOPE scores differ significantly across chunking strategies'
            }

        by_strategy = df.groupby('chunking_strategy')['hope_score'].mean()
        best_strategy = by_strategy.idxmax()
        worst_strategy = by_strategy.idxmin()
        
        best_scores = df[df['chunking_strategy'] == best_strategy]['hope_score']
        worst_scores = df[df['chunking_strategy'] == worst_strategy]['hope_score']
        
        if len(best_scores) > 1 and len(worst_scores) > 1:
            t_stat, p_value = stats.ttest_ind(best_scores, worst_scores, equal_var=False)
            tests['best_vs_worst'] = {
                'best_strategy': best_strategy,
                'worst_strategy': worst_strategy,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'mean_difference': float(best_scores.mean() - worst_scores.mean())
            }
        
        return tests
    
    def _identify_best_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:

        strategy_stats = df.groupby('chunking_strategy').agg({
            'hope_score': ['mean', 'std', 'count'],
            'answer_correctness': ['mean', 'std'],
            'num_passages': ['mean', 'std']
        }).round(4)
        
        best_hope = strategy_stats['hope_score']['mean'].idxmax()
        best_rag = strategy_stats['answer_correctness']['mean'].idxmax()

        hope_ranking = strategy_stats['hope_score']['mean'].sort_values(ascending=False)
        rag_ranking = strategy_stats['answer_correctness']['mean'].sort_values(ascending=False)
        
        return {
            'best_by_hope': best_hope,
            'best_by_rag': best_rag,
            'hope_ranking': hope_ranking.to_dict(),
            'rag_ranking': rag_ranking.to_dict(),
            'strategy_statistics': strategy_stats.to_dict()
        }
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):

        print(f"\nOVERALL SUMMARY:")
        print(f"Total experiments: {analysis['overall_summary']['total_experiments']}")
        print(f"Average HOPE score: {analysis['overall_summary']['avg_hope_score']:.3f} ± {analysis['overall_summary']['std_hope_score']:.3f}")
        print(f"Average RAG accuracy: {analysis['overall_summary']['avg_rag_accuracy']:.3f} ± {analysis['overall_summary']['std_rag_accuracy']:.3f}")
        
        # Best strategies
        print(f"\nBEST STRATEGIES:")
        print(f"By HOPE score: {analysis['best_strategies']['best_by_hope']}")
        print(f"By RAG accuracy: {analysis['best_strategies']['best_by_rag']}")
        
        print(f"\nSTRATEGY PERFORMANCE (HOPE scores):")
        for strategy, stats in analysis['by_strategy'].items():
            print(f"{strategy:20} HOPE: {stats['avg_hope_score']:.3f}  RAG: {stats['avg_rag_accuracy']:.3f}  Passages: {stats['avg_passages']:.1f}")
        
        print(f"\nKEY CORRELATIONS (Pearson):")
        hope_score_corrs = analysis['correlations']['hope_score']
        for rag_metric, corr_data in hope_score_corrs.items():
            if 'pearson' in corr_data:
                pearson = corr_data['pearson']['correlation']
                p_val = corr_data['pearson']['p_value']
                sig = "*" if corr_data['pearson']['significant'] else ""
                print(f"  HOPE ↔ {rag_metric:20} r = {pearson:.3f}{sig} (p = {p_val:.3f})")

        print(f"\nSTATISTICAL TESTS:")
        if 'anova_hope_scores' in analysis['statistical_tests']:
            anova = analysis['statistical_tests']['anova_hope_scores']
            sig = "*" if anova['significant'] else ""
            print(f"  ANOVA (HOPE across strategies): F = {anova['f_statistic']:.3f}, p = {anova['p_value']:.3f}{sig}")
        
        if 'best_vs_worst' in analysis['statistical_tests']:
            test = analysis['statistical_tests']['best_vs_worst']
            sig = "*" if test['significant'] else ""
            print(f"  t-test ({test['best_strategy']} vs {test['worst_strategy']}):")
            print(f"    t = {test['t_statistic']:.3f}, p = {test['p_value']:.3f}{sig}")
            print(f"    Mean difference: {test['mean_difference']:.3f}")
        
    
    def _generate_visualizations(self, df: pd.DataFrame, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='chunking_strategy', y='hope_score')
        plt.title('HOPE Scores by Chunking Strategy')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('HOPE Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hope_by_strategy.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='chunking_strategy', y='answer_correctness')
        plt.title('RAG Accuracy by Chunking Strategy')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Answer Correctness')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rag_by_strategy.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        hope_metrics = ['zeta_con', 'zeta_sem', 'zeta_inf', 'zeta_align', 'hope_score']
        rag_metrics = ['answer_correctness', 'response_relevancy', 'factual_correctness', 'context_recall']
        
        corr_matrix = df[hope_metrics + rag_metrics].corr()
        hope_rag_corr = corr_matrix.loc[hope_metrics, rag_metrics]
        
        sns.heatmap(hope_rag_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation: HOPE Metrics vs RAG Metrics')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='hope_score', y='answer_correctness', 
                        hue='chunking_strategy', style='chunking_strategy', s=100)
        
        z = np.polyfit(df['hope_score'], df['answer_correctness'], 1)
        p = np.poly1d(z)
        plt.plot(df['hope_score'], p(df['hope_score']), "r--", alpha=0.5, 
                label=f"r = {df['hope_score'].corr(df['answer_correctness']):.3f}")
        
        plt.title('HOPE Score vs RAG Accuracy')
        plt.xlabel('HOPE Score')
        plt.ylabel('Answer Correctness')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hope_vs_rag.png", dpi=300)
        plt.close()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        hope_components = ['zeta_con', 'zeta_sem', 'zeta_inf', 'zeta_align']
        titles = ['Concept Unity', 'Semantic Independence', 
                    'Information Preservation', 'Multimodal Alignment']
        
        for idx, (component, title) in enumerate(zip(hope_components, titles)):
            ax = axes[idx // 2, idx % 2]
            sns.boxplot(data=df, x='chunking_strategy', y=component, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('HOPE Component Scores by Chunking Strategy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hope_components.png", dpi=300)
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}/")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


def main():
    experiment = HOPEExperiment()

    data_dir = "documents"          
    results_dir = "test"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "hope_results.json")
    analysis_file = os.path.join(results_dir, "analysis.json")

    documents = read_documents(data_dir)
    
    print("HOPE 2.0 EXPERIMENT")
    print(f"Documents to process: {len(documents)}")
    print(f"Results will be saved to: {results_dir}/")
    
    results = experiment.run_experiment(documents, output_file)
    analysis = experiment.analyze_results(results, analysis_file)
    
    print(f"\nResults saved to:")
    print(f"Raw results: {output_file}")
    print(f"Analysis: {analysis_file}")

    summary_file = os.path.join(results_dir, "summary.md")
    with open(summary_file, 'w') as f:
        f.write("HOPE 2.0 Experiment Summary\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Documents: {len(documents)}\n")
        f.write(f"**Experiments**: {len(results)}\n\n")
        
        f.write("Key Findings\n\n")
        f.write(f"Average HOPE Score: {analysis['overall_summary']['avg_hope_score']:.3f} ± {analysis['overall_summary']['std_hope_score']:.3f}\n")
        f.write(f"Average RAG Accuracy: {analysis['overall_summary']['avg_rag_accuracy']:.3f} ± {analysis['overall_summary']['std_rag_accuracy']:.3f}\n")
        f.write(f"Best Strategy (HOPE): {analysis['best_strategies']['best_by_hope']}\n")
        f.write(f"Best Strategy (RAG): {analysis['best_strategies']['best_by_rag']}\n\n")
        
        f.write("Strategy Rankings\n\n")
        f.write("By HOPE Score:\n")
        for i, (strategy, score) in enumerate(analysis['best_strategies']['hope_ranking'].items(), 1):
            f.write(f"{i}. {strategy}: {score:.3f}\n")
        
        f.write("By RAG Accuracy:\n")
        for i, (strategy, score) in enumerate(analysis['best_strategies']['rag_ranking'].items(), 1):
            f.write(f"{i}. {strategy}: {score:.3f}\n")
    
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()