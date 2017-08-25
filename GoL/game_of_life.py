from itertools import product

class Game:

	def __init__(self, board):
		if len(board) != 400:
			raise Exception("Board size was not 400")
		self.board = board

	def neighbor_cnt(self, cell):
		count = 0
		x_range = xrange(cell[0] - 1, cell[0] + 2)
		y_range = xrange(cell[1] - 1, cell[1] + 2)
		neighbor_cells = product(x_range, y_range)
		for neighbor in neighbor_cells:
			count += self.life_status(neighbor)
		count -= self.life_status(cell)
		return count

	def	life_status(self, cell):
		if -1 < cell[0] < 20 and -1 < cell[1] < 20 and self.board[20 * cell[0] + cell[1]] == 1:
			return 1
		else:
			return 0
	
	def print_board(self):
		for i in xrange(20):
			for j in xrange(20):
				print(self.board[j * 20 + i], end='')
			print("")
	
	def advance(self):
		next_board = []
		for i in xrange(20):
			for j in xrange(20):
				cell = (i, j)
				neighbor_cnt = self.neighbor_cnt(cell)
				alive = self.life_status(cell)
				if alive and neighbor_cnt == 2 or neighbor_cnt == 3:
					next_board.append(1)
				elif not alive and neighbor_cnt == 3:
					next_board.append(1)
				else:
					next_board.append(0)
		self.board = next_board
