

from src.setup import *

from src.bias_scores.w_seat import WSEAT
from src.visualization import plot_results

import pandas as pd


if __name__ == "__main__":
    # test = "sent-weat6"    
    data = load_data_dict()
    all_models = get_all_models()
    all_tests = list(data.keys())
    

    all_results = []
    for model_name in all_models:
            logger.info(f"Running model {model_name}...")
            wseat_obj = WSEAT(model_name, n_samples=1000, parametric=False)
            for test in all_tests:
                logger.info(f"Running test {test}...")
                results = wseat_obj.run_wseat_test(data[test], test, report=False)
                all_results.append(results)

    results_df = pd.DataFrame(all_results)
    # results_df.to_csv("results.csv")
    # plot_results(results_df)

    # example
    # wseat = WSEAT("bert-base-uncased", n_samples=1000, parametric=False)
    # results = wseat.run_wseat_test(data[test], test, report=True)