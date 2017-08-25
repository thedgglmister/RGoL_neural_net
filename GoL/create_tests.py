import random
import csv
import game_of_life as gol

def create_tests(test_cnt, file_name):
	with open(file_name, "wb") as csv_file:
		csvwriter = csv.writer(csv_file)
		for i in range(test_cnt):
			print (i)
			delta = random.randint(1, 5)
			percent_alive = random.uniform(0.01, .99)
			alive_cnt = int(round(400 * percent_alive))
			while (True):
				init_board = [1] * alive_cnt + [0] * (400 - alive_cnt)
				random.shuffle(init_board)
				game = gol.Game(init_board)
				for j in range(5):
					game.advance()
				start_board = game.board
				for k in range(delta):
					game.advance()
				stop_board = game.board
				if (sum(stop_board)):
					break
			row = [i, delta] + start_board + stop_board
			csvwriter.writerow(row)
