import numpy as np 
import plotly.graph_objects as go
from blackjack import blackJack
from pathlib import Path

def policy(state) -> int:
    """A simple policy: stick if the sum is 20 or 21, otherwise hit."""
    return 0 if state.playerSum >= 20 else 1

# --------------------------------------------------------
## Edit this value to change number of episodes simulated (too lazy to add a cmd line arg)
numEpisodes = 100000
## --------------------------------------------------------

values = {}
# instead of keeping track of every single reward we have ever gotten for a state we can just keep track how many times we have visited a state 
# check out 2.4 Incremental Implementation
visits = {}


# Simulating episodes
for episode in range(numEpisodes):
    game = blackJack()
    states = [game.state.getStateTuple()]

    while not game.state.terminal:
        # This is essentially giving the state to the policy function and then passing that action back to the game and storing the resulting state.
        state = game.action(policy(game.state)).getStateTuple()
        # For first visit MC we only add the state if we have not seen it before in this episode
        if state not in states:
            states.append(state)

    for state in states:
        if visits.get(state) is None:
            visits[state] = 0
            values[state] = 0.0
        visits[state] += 1
        # from 2.4 Incremental Implementation
        values[state] += (game.reward - values[state])/visits[state]
    
    print(f'Episode {episode+1} complete')

# Generating surface plots. (This is probably ineffecient but oh well.)
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
y = [1,2,3,4,5,6,7,8,9,10]

outputPath = Path('./plots')
outputPath.mkdir(parents=True, exist_ok=True)

for usable in [True, False]:
    graph = []
    for p in x:
        arr = []
        for d in y:
            key = (usable, p, d)
            if not values.get(key) is None:
                arr.append(values.get(key))
            else:
                arr.append(0)
        graph.append(arr)

    fig = go.Figure(data=[go.Surface(z=graph, x=x, y=x)])
    fig.update_layout(title=f'State Vals for {"usable ace" if usable else "no ace"} with {numEpisodes} episodes', autosize=True,
                      scene=dict(
                          xaxis_title='Dealer Showing',
                          yaxis_title='Player Sum',
                          zaxis_title='State Value'),
                      margin=dict(l=65, r=50, b=65, t=90))
    name = f'blackjack_surface_{"ace" if usable else "no_ace"}_{numEpisodes}eps.html'
    fig.write_html(outputPath / name, include_plotlyjs='cdn')
    print(f'Wrote figure to {name}')
    