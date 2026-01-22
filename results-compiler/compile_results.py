from pathlib import Path
from experiment_compiler import ExperimentCompiler
import datetime

def main():
    # Initialize compiler
    
    #base_dir = Path("local_experiments/")
    base_dir = Path("terumo_experiments/att-metric")
    compiler = ExperimentCompiler(base_dir)
    
    # Compile results
    results = compiler.compile_results()
    
    # Save compiled results with timestamp in filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results['timestamp'] = timestamp
    compiler.save_results(results, f"results/compiled_results_{timestamp}.json")

if __name__ == "__main__":
    main()