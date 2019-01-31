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
mpl.rc('axes', facecolor='white',
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
        vectorizer = CountVectorizer(ngram_range=(1, 2),
                                     token_pattern=r'\b\w+\b')
    X = vectorizer.fit_transform(critics.quote)
    Y = (critics.fresh == 'fresh').values.astype(int)
    return X, Y
