
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    from src.setup import logger, RESULTS_DIR
except ModuleNotFoundError:
    from setup import logger, RESULTS_DIR


def plot_results(df):    
    # remove heilman tests
    df = df[df["test"].str.contains("heilman") == False]    

    df.groupby(["model", "test"]).mean().reset_index().pivot(
        index="model", 
        columns="test", 
        values="effect_size"
    ).plot(
        kind="bar",
        figsize=(13, 6),
        title="Effect Size by Model and Test",
        ylabel="Effect Size",
        xlabel="Model",
        rot=0,
        colormap="Set2" 
        # colormap="Accent"
    )
    
    plt.legend(
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fancybox=False,
        shadow=False,
        frameon=False,
        edgecolor=None

    )
    plt.subplots_adjust(
        left=0.1,
        right=0.75
    )

    # add horizontal line at 0
    plt.axhline(
        y=0, 
        color='black', 
        linestyle='-', 
        linewidth=0.5
    )

    # add grid lines for y axis only
    plt.grid(
        axis='y',
        color='gray',
        linestyle='-',
        linewidth=0.5,
        alpha=0.5                
    )

    plt.savefig(f"{RESULTS_DIR}results.png", bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv(f"{RESULTS_DIR}results_v1.csv")
    plot_results(df)