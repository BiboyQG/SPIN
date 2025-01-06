from collections import OrderedDict
from deepdiff import DeepDiff
from fuzzywuzzy import fuzz
from pathlib import Path
import json
import csv


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
    diff = DeepDiff(
        ground_truth,
        test_object,
        ignore_order=True,
        exclude_paths=[
            "root['strengths']",
            "root['weaknesses']",
            "root['overallVerdict']",
        ],
        ignore_type_in_groups=[(int, float)],
        ignore_numeric_type_changes=True,
    )
    results["value_exactness"] = 1 - len(diff) / (len(ground_truth) + len(test_object))

    # Numeric value similarity
    def compare_numeric(gt_val, test_val):
        if isinstance(gt_val, (int, float)) and isinstance(test_val, (int, float)):
            return 1 - min(abs(gt_val - test_val) / max(abs(gt_val), 1), 1)
        return 0

    # String similarity
    def compare_string(gt_val, test_val):
        if isinstance(gt_val, str) and isinstance(test_val, str):
            # Use ratio for shorter strings
            if len(gt_val) < 10 and len(test_val) < 10:
                return fuzz.ratio(gt_val, test_val) / 100
            # Use token_sort_ratio for longer strings
            else:
                return fuzz.token_sort_ratio(gt_val, test_val) / 100
        return 0

    numeric_similarity = []
    string_similarity = []

    def recursive_compare(gt_obj, test_obj):
        if isinstance(gt_obj, dict) and isinstance(test_obj, dict):
            for key in gt_obj:
                if key in test_obj:
                    if key in ["strengths", "weaknesses", "overallVerdict"]:
                        continue
                    recursive_compare(gt_obj[key], test_obj[key])
        elif isinstance(gt_obj, list) and isinstance(test_obj, list):
            for gt_item, test_item in zip(gt_obj, test_obj):
                recursive_compare(gt_item, test_item)
        else:
            if isinstance(gt_obj, (int, float)) and isinstance(test_obj, (int, float)):
                numeric_similarity.append(compare_numeric(gt_obj, test_obj))
            elif isinstance(gt_obj, str) and isinstance(test_obj, str):
                string_similarity.append(compare_string(gt_obj, test_obj))

    recursive_compare(ground_truth, test_object)

    results["numeric_similarity"] = (
        sum(numeric_similarity) / len(numeric_similarity) if numeric_similarity else 1
    )
    results["string_similarity"] = (
        sum(string_similarity) / len(string_similarity) if string_similarity else 1
    )

    return results


def get_schemas():
    """Get all available schemas from the ground truth directory."""
    gt_path = Path("results/gt")
    return [schema.name for schema in gt_path.iterdir() if schema.is_dir()]


def get_model_paths():
    """Get paths of all available models."""
    results_path = Path("results")
    # Exclude special directories
    excluded_dirs = {"gt", "scrape", "fsm"}
    return [
        model_path
        for model_path in results_path.iterdir()
        if model_path.is_dir() and model_path.name not in excluded_dirs
    ]


def get_depth_levels(model_path, schema):
    """Get all available depth levels for a given model and schema."""
    schema_path = model_path / schema
    if not schema_path.exists():
        return []
    return [depth.name for depth in schema_path.iterdir() if depth.is_dir()]


def evaluate_models_new():
    results = []
    schemas = get_schemas()
    model_paths = get_model_paths()

    for schema in schemas:
        print(f"Processing schema: {schema}")
        gt_path = Path("results/gt") / schema
        gt_files = list(gt_path.glob("*.json"))
        gt_files.sort(key=lambda x: x.stem)

        for model_path in model_paths:
            model_name = model_path.name
            print(f"Processing model: {model_name}")

            depth_levels = get_depth_levels(model_path, schema)

            for depth in depth_levels:
                depth_results = []
                print(f"Processing depth: {depth}")

                for gt_file in gt_files:
                    test_file = model_path / schema / depth / gt_file.name

                    try:
                        with gt_file.open("r") as f:
                            ground_truth = json.load(f)

                        if test_file.exists():
                            with test_file.open("r") as f:
                                test_object = json.load(f)

                            result = compare_json_objects(ground_truth, test_object)
                            result.update(
                                {
                                    "sample_no": gt_file.stem,
                                    "model_name": model_name,
                                    "schema": schema,
                                    "depth": depth,
                                }
                            )
                            depth_results.append(result)
                        else:
                            print(f"Missing test file: {test_file}")

                    except Exception as e:
                        print(f"Error processing {gt_file}: {str(e)}")
                        continue

                # Calculate averages for this depth level
                if depth_results:
                    avg_result = {
                        "sample_no": "avg",
                        "model_name": model_name,
                        "schema": schema,
                        "depth": depth,
                        "json_validity": sum(r["json_validity"] for r in depth_results)
                        / len(depth_results),
                        "key_similarity": sum(
                            r["key_similarity"] for r in depth_results
                        )
                        / len(depth_results),
                        "value_exactness": sum(
                            r["value_exactness"] for r in depth_results
                        )
                        / len(depth_results),
                        "numeric_similarity": sum(
                            r["numeric_similarity"] for r in depth_results
                        )
                        / len(depth_results),
                        "string_similarity": sum(
                            r["string_similarity"] for r in depth_results
                        )
                        / len(depth_results),
                    }
                    results.extend(depth_results + [avg_result])

    # Calculate overall model averages
    model_averages = {}
    for model_path in model_paths:
        model_name = model_path.name
        model_results = [
            r
            for r in results
            if r["model_name"] == model_name and r["sample_no"] == "avg"
        ]

        if model_results:
            model_averages[model_name] = {
                "sample_no": "overall_avg",
                "model_name": model_name,
                "schema": "all",
                "depth": "all",
                "json_validity": sum(r["json_validity"] for r in model_results)
                / len(model_results),
                "key_similarity": sum(r["key_similarity"] for r in model_results)
                / len(model_results),
                "value_exactness": sum(r["value_exactness"] for r in model_results)
                / len(model_results),
                "numeric_similarity": sum(
                    r["numeric_similarity"] for r in model_results
                )
                / len(model_results),
                "string_similarity": sum(r["string_similarity"] for r in model_results)
                / len(model_results),
            }

    results.extend(model_averages.values())
    return results


def save_results_to_csv(results, filename="evaluation_results.csv"):
    """Save results to CSV with the new structure."""
    if not results:
        print("No results to save. The results list is empty.")
        return

    fieldnames = [
        "sample_no",
        "model_name",
        "schema",
        "depth",
        "json_validity",
        "key_similarity",
        "value_exactness",
        "numeric_similarity",
        "string_similarity",
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = evaluate_models_new()
    save_results_to_csv(results, "results/comprehensive_evaluation_results.csv")
