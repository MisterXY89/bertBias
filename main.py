

from src.setup import *

from src.bias_scores.w_seat import WSEAT
from src.visualization import plot_results

import argparse
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help=f"model name, available models: { ','.join(get_all_models()) }")
    parser.add_argument("--test", type=str, default="all", help=f"test name, available tests: { ','.join(list(load_data_dict().keys())) }")

    args = parser.parse_args()
    model_name = args.model
    test = args.test

    data = load_data_dict()
    all_models = get_all_models()
    all_tests = list(data.keys())

    if model_name != "all":
        all_models = [model_name]
    if test != "all":
        all_tests = [test]    

    all_results = []
    for model_name in all_models:
            logger.info(f"Running model {model_name}...")
            wseat_obj = WSEAT(model_name, n_samples=1000, parametric=False)
            for test in all_tests:
                logger.info(f"Running test {test}...")
                results = wseat_obj.run_wseat_test(data[test], test, report=False)
                all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}results.csv")    
    plot_results(results_df)

    # example
    # wseat = WSEAT("bert-base-uncased", n_samples=1000, parametric=False)
    # results = wseat.run_wseat_test(data[test], test, report=True)