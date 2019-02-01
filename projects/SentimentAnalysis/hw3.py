import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.rcsetup import cycler
from sklearn.feature_extraction.text import CountVectorizer


dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]


from matplotlib.rcsetup import cycler

mpl.rc('figure', figsize=(10,6), dpi=150)
mpl.rc('axes', facecolor='white', labelsize='medium',
       prop_cycle=cycler('color', dark2_colors ))
mpl.rc('lines', lw=2)
mpl.rc('patch', ec='white', fc=dark2_colors[0])
mpl.rc('font', size=14, family='StixGeneral')


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


def histogram_style():
    remove_border(left=False)
    plt.grid(False)
    plt.grid(axis='y', color='w', linestyle='-', lw=1)


def make_xy(critics, vectorizer=None):
    """
    Build a bag-of-words training set for the review data
    - args
        critics(dataFrame) : The review data

        vectorizer : CountVectorizer object (optional)
        A CountVectorizer object to use. If None,
        then create and fit a new CountVectorizer.
        Otherwise, re-fit the provided CountVectorizer
        using the critics data

    -returns
        X : numpy array (dims: nreview, nwords)
            Bag-of-words representation for each review.
        Y : numpy array (dims: nreview)
            1/0 array. 1 = fresh review, 0 = rotten review

    Examples
    --------
    X, Y = make_xy(critics)
    """
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(critics.quote)
    Y = (critics.fresh == 'fresh').values.astype(int)
    return X, Y


def calibration_plot(clf, xtest, ytest):
  '''
  Plots model-estimated probabilities against observed probabilities,
  to determine whether the model ove/underfits, how & to what extent.
  args
  -------
  clf : Classifier object
      A MultinomialNB classifier
  X : (Nexample, Nfeature) array
      The bag-of-words data
  Y : (Nexample) integer array
      1 if a review is Fresh
  '''
  # model probabilities of freshness
  prob = clf.predict_proba(xtest)[:,1]
  # observed responces
  outcome = ytest
  data = pd.DataFrame(dict(prob=prob, outcome=outcome))

  bins = np.linspace(0,1,20)
  # sort probs & group into 20 equal-sized bins
  cuts = pd.cut(prob, bins)
  binwidth = bins[1] - bins[0]

  # compute average observed freshness per bin
  cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
  # pmid is the average model probability of each bin
  cal['pmid'] = (bins[:-1] + bins[1:]) / 2
  # sig is the model uncertainty (avg error) for each bin
  cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])

  # setup subplots
  fig, axes = plt.subplots(2,1, sharex=False, figsize=(11,9))

  axes[0].set_title('Model Calibration')

  # plot model estimations vs observed probs with errors
  axes[0].errorbar(cal.pmid, cal['mean'], cal['sig'],
                   elinewidth=1.8, capsize=2.5)
  # plot y=x as perfect-fit reference
  axes[0].plot(cal.pmid, cal.pmid,
               linestyle='--', lw=1,
               color='k')

  axes[1].hist(prob, bins=bins,
                  log=True, rwidth=0.8)

  axes[0].set(xlabel="Predicted ~ P(Fresh)",
              ylabel="Empirical ~ P(Fresh)")

  axes[1].set(xlabel="Predicted ~ P(Fresh)",
             ylabel="Numbers - log scale")

