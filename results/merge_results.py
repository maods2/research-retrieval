import json


benchmark_path = "results/compiled_results_benchmark_20251103_162257.json"
with open(benchmark_path, "r") as f:
    benchmark_results = json.load(f)    

attention_metric_path = "results/compiled_results_attention_metric_20260122_013254.json"
with open(attention_metric_path, "r") as f:
    attention_metric_results = json.load(f)
    
print(benchmark_results.keys())
print(attention_metric_results.keys())


copy_dict = benchmark_results.copy()

for dataset in copy_dict.keys():
    if dataset == 'timestamp':
        continue
    
    for model in attention_metric_results[dataset]["models"].keys():
        copy_dict[dataset]["models"][model] = attention_metric_results[dataset]["models"][model].copy()
        
output_path = "results/merged_results_20260122_013254.json"
with open(output_path, "w") as f:
    json.dump(copy_dict, f, indent=4)
    
    