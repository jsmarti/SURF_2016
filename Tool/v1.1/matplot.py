"""
Initiallization of matplotlib for paper figures.

Author:
    Ilias Bilionis

Date:
    8/18/2015

"""


import matplotlib
matplotlib.use('PS')
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set_style('darkgrid')


def get_smallest_fig():
    fig, ax = plt.subplots(figsize=(1.54, 1.54 / 1.3))
    fig.subplots_adjust(left=0.15, bottom=0.15)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    return fig, ax


def get_column_fig(will_have_labels=True):
    fig, ax = plt.subplots(figsize=(3.31, 3.31 / 1.3))
    if will_have_labels:
        fig.subplots_adjust(left=0.15, bottom=0.15)
    return fig, ax


def get_two_column_fig():
    fig, ax = plt.subplots(figsize=(5.01, 5.01 / 1.3))
    return fig, ax


def get_full_fig(return_ax=True):
    fig = plt.figure(figsize=(6.85, 6.85 / 1.3))
    if return_ax:
        ax = fig.add_subplot(111)
        return fig, ax
    return fig
