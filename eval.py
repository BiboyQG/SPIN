from collections import OrderedDict
from deepdiff import DeepDiff
from fuzzywuzzy import fuzz
import json
import csv
import os


def compare_json_objects(ground_truth, test_object):
    results = OrderedDict()

    # 1. Structural Accuracy

    # JSON validity
    try:
        json.loads(json.dumps(test_object))
        results["json_validity"] = True
    except json.JSONDecodeError:
        results["json_validity"] = False

    # Key similarity
    gt_keys = set(ground_truth.keys())
    test_keys = set(test_object.keys())
    results["key_similarity"] = len(gt_keys.intersection(test_keys)) / len(
        gt_keys.union(test_keys)
    )

    # 2. Content Accuracy

    # Value exactness
    diff = DeepDiff(ground_truth, test_object, ignore_order=True)
    results["value_exactness"] = 1 - len(diff) / (len(ground_truth) + len(test_object))

    # Numeric value similarity
    def compare_numeric(gt_val, test_val):
        if isinstance(gt_val, (int, float)) and isinstance(test_val, (int, float)):
            return 1 - min(abs(gt_val - test_val) / max(abs(gt_val), 1), 1)
        return 0

    # String similarity
    def compare_string(gt_val, test_val):
        if isinstance(gt_val, str) and isinstance(test_val, str):
            return fuzz.ratio(gt_val, test_val) / 100
        return 0

    numeric_similarity = []
    string_similarity = []

    def recursive_compare(gt_obj, test_obj):
        if isinstance(gt_obj, dict) and isinstance(test_obj, dict):
            for key in gt_obj:
                if key in test_obj:
                    recursive_compare(gt_obj[key], test_obj[key])
        elif isinstance(gt_obj, list) and isinstance(test_obj, list):
            for gt_item, test_item in zip(gt_obj, test_obj):
                recursive_compare(gt_item, test_item)
        else:
            numeric_similarity.append(compare_numeric(gt_obj, test_obj))
            string_similarity.append(compare_string(gt_obj, test_obj))

    recursive_compare(ground_truth, test_object)

    results["numeric_similarity"] = (
        sum(numeric_similarity) / len(numeric_similarity) if numeric_similarity else 1
    )
    results["string_similarity"] = (
        sum(string_similarity) / len(string_similarity) if string_similarity else 1
    )

    return results


def evaluate_models(ground_truth_dir, open_source_dir, proprietary_dir):
    results = []
    json_files = [f for f in os.listdir(ground_truth_dir) if f.endswith(".json")]
    json_files.sort(key=lambda x: int(x.split(".")[0]))

    for filename in json_files:
        ground_truth_path = os.path.join(ground_truth_dir, filename)
        print(f"Processing ground truth file: {ground_truth_path}")
        
        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
        except Exception as e:
            print(f"Error reading ground truth file {filename}: {str(e)}")
            continue

        # Evaluate open-source models
        print(f"Checking open-source models for {filename}")
        for model_dir in os.listdir(open_source_dir):
            model_path = os.path.join(open_source_dir, model_dir, 'instructor', filename)
            print(f"Checking path: {model_path}")
            if os.path.exists(model_path):
                print(f"Processing open-source model: {model_path}")
                try:
                    with open(model_path, 'r') as f:
                        test_object = json.load(f)
                    result = compare_json_objects(ground_truth, test_object)
                    result['sample no.'] = filename.split('.')[0]
                    result['model_name'] = "Qwen2.5-72B"
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {model_path}: {str(e)}")
            else:
                print(f"File not found: {model_path}")

        # Evaluate proprietary models
        print(f"Checking proprietary models for {filename}")
        for model_dir in os.listdir(proprietary_dir):
            model_path = os.path.join(proprietary_dir, model_dir, filename)
            print(f"Checking path: {model_path}")
            if os.path.exists(model_path):
                print(f"Processing proprietary model: {model_path}")
                try:
                    with open(model_path, 'r') as f:
                        test_object = json.load(f)
                    result = compare_json_objects(ground_truth, test_object)
                    result['sample no.'] = filename.split('.')[0]
                    result["model_name"] = model_dir
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {model_path}: {str(e)}")
            else:
                print(f"File not found: {model_path}")

    print(f"Total results processed: {len(results)}")
    return results


def save_results_to_csv(results, filename="evaluation_results.csv"):
    if not results:
        print("No results to save. The results list is empty.")
        return

    # Get all unique keys from all result dictionaries
    all_keys = ["json_validity", "key_similarity", "value_exactness", "numeric_similarity", "string_similarity"]

    # Define the order of columns, ensuring all keys are included
    fieldnames = ["sample no.", "model_name"] + [
        key for key in all_keys if key not in ["model_name", "sample no."]
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {filename}")


# Example usage
ground_truth_dir = 'dataset/results/gt'
open_source_dir = 'dataset/results/open-source/Qwen'
proprietary_dir = 'dataset/results/proprietary'

print(f"Ground truth directory: {ground_truth_dir}")
print(f"Open-source directory: {open_source_dir}")
print(f"Proprietary directory: {proprietary_dir}")

print(f"Open-source directory exists: {os.path.exists(open_source_dir)}")
print(f"Proprietary directory exists: {os.path.exists(proprietary_dir)}")

if os.path.exists(open_source_dir):
    print("Open-source models:", os.listdir(open_source_dir))
if os.path.exists(proprietary_dir):
    print("Proprietary models:", os.listdir(proprietary_dir))

results = evaluate_models(ground_truth_dir, open_source_dir, proprietary_dir)
save_results_to_csv(results)

# Print results to console as well
for result in results:
    print(json.dumps(result, indent=2))
