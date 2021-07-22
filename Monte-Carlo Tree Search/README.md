## Python implementation of Monte Carlo Tree Search algorithm

 
Basic python implementation of [Monte Carlo Tree Search](https://int8.io/monte-carlo-tree-search-beginners-guide) (MCTS) intended to run on small game trees. 

### Running tic-tac-toe example 

Run example:

```python

import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe import TicTacToeGameState

state = np.zeros((3,3))
initial_board_state = TicTacToeGameState(state = state, next_to_move=1)

root = TwoPlayersGameMonteCarloTreeSearchNode(state = initial_board_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(10000)

```


### 蒙特卡洛树搜索的一些介绍：

#### 详细全面的介绍蒙特卡洛树搜索
http://repository.essex.ac.uk/4117/1/MCTS-Survey.pdf

