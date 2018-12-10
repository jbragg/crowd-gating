"""Module for various pylot utils."""

import numpy as np
import matplotlib.pyplot as plt


def histogram_plotter(ax, x_labels, y_labels, width=0.8):
    """Plot histogram on the given axes."""
    n = len(x_labels)
    tickLocations = np.arange(n)
    rectLocations = tickLocations - (width / 2.0)
    rects = ax.bar(rectLocations,
           y_labels,
           width)
    label_rect(ax, rects)
    ax.set_xticks(ticks=tickLocations)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(min(tickLocations) - 0.6, max(tickLocations) + 0.6)
    ax.set_yticks(np.arange(y_labels.min(), y_labels.max(), int(y_labels.std())))
    ax.yaxis.grid(True)


def label_rect(ax, rects):
    """Attach text to bar chart rectangles."""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom', rotation=90)


def savefig(ax, name):
    """Save the current figure taking into account legends."""
    lgd = ax.get_legend()
    if lgd is not None:
        plt.savefig(name, bbox_inches='tight', bbox_extra_artists=(lgd,))
    else:
        plt.savefig(name, bbox_inches='tight')


def tsplot_robust(df, time, unit, condition, value, ci=95,
                  logx=False, logy=False):
    """Plot timeseries data with different x measurements.

    Returns:
        ax:     Axis object.
        df:     Dataframe containing: time, condition, mean, n, sem.

    """
    import pandas as pd
    import seaborn as sns
    import scipy.stats as ss
    n = df.groupby([condition, time])[value].count()
    n.name = 'n'
    means = df.groupby([condition, time])[value].mean()
    means.name = 'mean'
    sem = df.groupby([condition, time])[value].aggregate(ss.sem)
    sem.name = 'sem'
    df_stat = pd.concat([means, n, sem], axis=1).reset_index()

    # Use seaborn iff all conditions have the same number of measurements for
    # each time point.
    n_pivot = n.reset_index().pivot(index=time, columns=condition, values='n')
    if len(df) == 0:
        raise Exception('Unable to plot empty dataframe')
    if len(n_pivot) == 1:
        ax = sns.barplot(condition, y=value, data=df, ci=ci)
    elif len(n_pivot) == sum(n_pivot.duplicated()) + 1:
        ax = sns.tsplot(df, time=time, condition=condition,
                        unit=unit, value=value, ci=ci)
    else:
        if ci != 95:
            raise NotImplementedError
        ax = plt.gca()
        for c, df in df_stat.groupby(condition):
            line, = plt.plot(df[time], df['mean'], label=c)
            # Disable filling for logy, since may exceed desirable range.
            if not logy:
                ax.fill_between(df[time],
                                df['mean'] - 1.96 * df['sem'],
                                df['mean'] + 1.96 * df['sem'],
                                facecolor=line.get_color(),
                                alpha=0.5,
                                where=np.isfinite(df['sem']))
        plt.legend()
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    return ax, df_stat
