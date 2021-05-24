import argparse

import matplotlib.pyplot as plt
import numpy as np
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    category_names = ['Not working', 'Working for constant wind only','Working']

    results = { #'During training without PSF': [10,25],
                'After training without PSF': [0,10,20],
                'During training with PSF': [13,0,25],
                'After training with PSF': [13,0,25]
            }
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(5, np.max(data))
    ax.grid(axis='x')
    ax.set_axisbelow(True)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        xcenters = starts + widths / 2

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    ax.set_xticks(np.arange(5, np.max(data)+1,1))
    ax.set_xlabel('Wind speed [m/s]')

    if args.save:
        plt.savefig('plots/results_overview_bar.pdf', bbox_inches='tight')

    plt.show()