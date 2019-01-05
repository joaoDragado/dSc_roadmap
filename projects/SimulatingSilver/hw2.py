from collections import defaultdict
import json

import numpy as np
import pandas as pd
from scipy.special import erf

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

#this mapping between states and abbreviations will come in handy later
states_abbrev = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


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

#load in state geometry
state2poly = defaultdict(list)

with open("../data/hw2_data/us-states.json") as json_data:
    data = json.load(json_data)
    for f in data['features']:
        state = states_abbrev[f['id']]
        geo = f['geometry']
        if geo['type'] == 'Polygon':
            for coords in geo['coordinates']:
                state2poly[state].append(coords)
        elif geo['type'] == 'MultiPolygon':
            for polygon in geo['coordinates']:
                state2poly[state].extend(polygon)


def draw_state(plot, stateid, **kwargs):
    """
    draw_state(plot, stateid, color=..., **kwargs)

    Automatically draws a filled shape representing the state in
    subplot.
    The color keyword argument specifies the fill color.  It accepts keyword
    arguments that plot() accepts
    """
    for polygon in state2poly[stateid]:
        xs, ys = zip(*polygon)
        plot.fill(xs, ys, **kwargs)


def make_map(states, label):
  """
  Draw a cloropleth map, that maps data onto the United States
  Inputs
  -------
  states : Column of a DataFrame
      The value for each state, to display on a map
  label : str
      Label of the color bar
  Returns
  --------
  The map
  """
  fig = plt.figure(figsize=(12, 9))
  ax = plt.gca()
  if states.max() < 2: # colormap for election probabilities
      cmap = cm.RdBu
      vmin, vmax = 0, 1
  else:  # colormap for electoral votes
      cmap = cm.binary
      vmin, vmax = 0, states.max()
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
  skip = set(['National', 'District of Columbia', 'Guam', 'Puerto Rico',
              'Virgin Islands', 'American Samoa', 'Northern Mariana Islands'])
  for state in states_abbrev.values():
      if state in skip:
          continue
      color = cmap(norm(states.ix[state]))
      draw_state(ax, state, fc=color, ec='k')
  #add an inset colorbar
  ax1 = fig.add_axes([0.45, 0.70, 0.4, 0.02])
  cb1=mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
  ax1.set_title(label)
  remove_border(ax, left=False, bottom=False)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim(-180, -60)
  ax.set_ylim(15, 75)
  return ax



def simulate_election(model, n_sim):
    '''
    args :
      model (df) : summarizes an election forecast ; 51 rows, indexed by state.
        cols: Obama prob Obama wins the state,
        Votes : Electoral votes awarded
      n_sim (int) : Number of simulations to run
    returns : Numpy array with n_sim elements, each holds no of votes Obama wins.
    '''
    # each col simulates a single outcome from the 50 states + DC
    # Obama wins the simulation if
    # the random number is < the win probability
    simulations = np.random.uniform(size=(51, n_sim))
    obama_votes = (simulations <
                   model.Obama.values.reshape(-1, 1)
                  ) * model.Votes.values.reshape(-1, 1)
    # summing over rows gives the total electoral votes
    # for each simulation
    return obama_votes.sum(axis=0)


def plot_simulation(simulation):
    '''
    args :
      simulation: mumpy array with n_sim (see simulate_election) elements
      Each element stores the number of electoral college votes
      Obama wins in each simulation.
    '''
    plt.hist(simulation, bins=np.arange(240, 375, 1),
             label='simulations', align='left', density=True)
    plt.axvline(332, 0, .5, color='r',
                linestyle='--', label='Actual Outcome')
    plt.axvline(269, 0, .5, color='k',
                linestyle='--', label='Victory Threshold')

    p05 = np.percentile(simulation, 5.)
    p95 = np.percentile(simulation, 95.)
    iq = int(p95 - p05)

    pwin = ((simulation >= 269).mean() * 100)

    plt.title("Chance of Obama Victory: {:.1f}%, Spread: {} votes".format(
    pwin, iq))
    plt.legend(frameon=False, loc='upper left')
    plt.xlabel("Obama Electoral College Votes")
    plt.ylabel("Probability")
    remove_border()


def simple_gallup_model(gallup):
    """
    A simple forecast that predicts an Obama (Democratic) victory with
    0 or 100% probability, depending on whether a state
    leans Republican or Democrat.
    ------
    args :
        gallup (DataFrame) : The Gallup dataframe above

    returns :
        model (DataFrame) :
        cols :
            Obama: prob state votes for Obama. binary vals (0 or 1)

        model.index should be set to gallup.index (indexed by state name)
    """
    data = ((gallup.Dem_Adv > 0).astype(float))
    return pd.DataFrame(data=data).rename(columns={'Dem_Adv':'Obama'})


def uncertain_gallup_model(gallup):
    """
    predicts an Obama (Dem) victory if the random variable drawn
    from a Gaussian with mean Dem_Adv and standard deviation 3% is >0
    -------
    args :
        gallup (DataFrame) : The Gallup dataframe above

    returns :
        model (DataFrame) :
            cols : * Obama: probability that the state votes for Obama.
        model.index should be set to gallup.index (indexed by state name)
    """
    sigma = 3
    prob = 1 -  .5 * (1 + erf(-1*gallup.Dem_Adv / np.sqrt(2 * sigma**2)))

    return pd.DataFrame(data=prob, index=gallup.index).rename(columns={'Dem_Adv':'Obama'})


def biased_gallup(gallup, bias):
    """
    Simulates correcting a hypothetical bias towards Democrats
    in the original Gallup data.
    Subtracts a fixed amount from Dem_Adv,
    before computing the uncertain_gallup_model.
    -------
    args :
        gallup (DataFrame) : The Gallup party affiliation dataframe
        bias (float) : The amount by which to shift each prediction

    returns :
            model (DataFrame) :
                cols : * Obama: probability that the state votes for Obama.
    --Test--
    >>> model = biased_gallup(gallup, 1.)
    >>> model.ix['Florida']
    >>> .460172
    """
    g2 = gallup.copy()
    g2.Dem_Adv -= bias
    return uncertain_gallup_model(g2)
