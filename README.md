# Reverse Game of Life Neural Network
A Machine Learning project at 42 Silicon Valley

### --Project Description--
The goal of Reverse Game of Life is to use machine learning techniques to accurately predict a past board state of Conway's Game of Life, given any current board state. My approach was to build a feed forward neural network module (without use external libraries), as well as a parallel processing grid search algorithm to track results and fine-tune the hyperparameters of the network and prevent overfitting to the training data. 

## --The Game of Life--
The universe of the Game of Life is an infinite two-dimensional orthogonal grid of square cells, each of which is in one of two possible states, alive or dead. Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent. At each step in time, the following transitions occur:

  Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
  Any live cell with two or three live neighbours lives on to the next generation.
  Any live cell with more than three live neighbours dies, as if by overpopulation.
  Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

## --My Network--
I built a complete neural network class that can be re-purposed for other machine learning problems. It is a feed-forward network that can initialized with a custom size and cost functions, and it uses a stochastic gradient descent learning algorithm that takes a learning rate, regularization coefficient, batch-size, and epoch count as parameters. It also can be provided with up to two evaluation data sets and will display and log the cost and accuracy on these data sets after each epoch of training.

## --Results--

I used parallel processing and a grid search algorithm to find optimal hyperparameters. My best network contained a single hidden layer of 600 neurons, and had a learning rate of 0.01, regularization coefficient of 5.0, batch size of 10. After training with 200,000 data pieces for 30 epochs, the network was making predictions with about 87% accuracy.

<img src="/images/training_log.png" width="600">
