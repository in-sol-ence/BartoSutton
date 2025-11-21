# Examples and Solutions for ([Original Book](https://www.amazon.co.jp/exec/obidos/ASIN/0262039249/hatena-blog-22/) by Richard S. Sutton, Andrew G. Barto)

## Answers to the non-programming questions

To check my answers to the non programming questions I have been using this repo: [Reinforcement Learning 2nd Edition by Sutton Exercise Solutions](https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)

## Chapter 2 

## Chapter 4

## Chapter 5
### Example 1: Black Jack
#### How to run
1. In `chapt5/example1.py`, edit the variable `numEpisodes` to however many episodes you want to simulate.
2. After episodes are simulated two plots, (for usable and non-usable ace) for state values will show up in `chapt5/plots`.
3. To view the plots run the command `python -m http.server {port number}` and then open up localhost on your browser for that port number and navigate to the plots directory.
4. If you just want to view the plots, there is already 10,000,000 episodes and 1,000 episodes plots.
**Extra**: You can create graphs for your own policies by editing the policy function. 

#### Explaining the Plot

Notice how the plot doesn't look like the textbook's. This is because the textbook assumes any reasonable policy would not hit before the player reaches 12; hence, finding state values for anything under 12 is unnecessary. 
> Thus, the player makes decisions on the basis of three variables: his current sum (12–21), the dealer’s one showing card (ace–10), and whether or not he holds a usable ace.

If you look at the graph only for sums 12 to 21, it is similar to the textbook's.
