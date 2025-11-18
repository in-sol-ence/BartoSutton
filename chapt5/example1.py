import plotly.graph_objects as go
from blackjack import blackJack
from pathlib import Path

def policy(state) -> int:
    """A simple policy: stick if the sum is 20 or 21, otherwise hit."""
    return 0 if state.playerSum >= 20 else 1

# --------------------------------------------------------
## Edit this value to change number of episodes simulated (too lazy to add a cmd line arg)
numEpisodes = 1000000000
## --------------------------------------------------------

values = {}
# instead of keeping track of every single reward we have ever gotten for a state we can just keep track how many times we have visited a state 
# check out 2.4 Incremental Implementation
visits = {}


# Simulating episodes
for episode in range(numEpisodes):
    game = blackJack()

    while not game.state.terminal:
        # This is essentially giving the state to the policy function and then passing that action back to the game
        game.action(policy(game.state))

    checking = False
    for state in game.state.getStateTuples():
        # if state[1] == 11 or checking:
        #     print(state)
        #     checking = True
        if visits.get(state) is None:
            visits[state] = 0
            values[state] = 0.0
        visits[state] += 1
        # from 2.4 Incremental Implementation
        values[state] += (game.reward - values[state])/visits[state]
        if checking:
            print(f'Value of {state} updated to {values[state]}')
    
    if checking:
        print(f'Episode {episode+1} complete')

# Generating surface plots. (Converting the dict to an array like this is probably ineffecient but oh well.)

outputPath = Path('./plots')
outputPath.mkdir(parents=True, exist_ok=True)

for usable in [True, False]:
    graph = []
    for p in range(1, 22):
        arr = []
        for d in range(1, 11):
            key = (usable, p, d)
            if not values.get(key) is None:
                arr.append(values.get(key))
            else:
                arr.append(0)
        graph.append(arr)
    x = list(range(1, 11))
    y = list(range(1, 22))
    fig = go.Figure(data=[go.Surface(z=graph, x=x, y=y)])
    fig.update_layout(title=f'State Vals for {"usable ace" if usable else "no ace"} with {numEpisodes} episodes', autosize=True,
                      scene=dict(
                          xaxis_title='Dealer Showing',
                          yaxis_title='Player Sum',
                          zaxis_title='State Value'),
                      margin=dict(l=65, r=50, b=65, t=90))
    name = f'blackjack_surface_{"ace" if usable else "no_ace"}_{numEpisodes}eps.html'
    fig.write_html(outputPath / name, include_plotlyjs='cdn')
    print(f'Wrote figure to {name}')