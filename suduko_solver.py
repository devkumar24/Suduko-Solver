import numpy as np


def read_grid(filename = None):
        file = open(filename,"r")
        for line in file:
            pass
        return line
def create_grid(line):
    grid = np.zeros((9,9))
    n = 0 
    for i in range(9):
        for j in range(9):
            if line[n] != '.':
                grid[i][j] = line[n]
            else:
                grid[i][j] = -1
            n += 1
    return grid

class Suduko:
    def __init__(self,filename):
        self.filename = filename
        self.line = read_grid(self.filename)
        self.grid = create_grid(self.line)
        self.grid = self.grid.astype('int64')
        self.solved_grid = self.grid

    def showGrid(self):
        grid = self.grid
        if type(grid) == np.ndarray:
            h,w = grid.shape
            if h == 9 and w == 9:
                print(grid)

            else:
                raise KeyError("Shape of Grid (9,9)! = ({},{})".format(h,w))

        else:
            if type(grid) == list:
                if len(grid) == 81 or len(grid) == 9:
                    grid = np.array(grid)
                    grid = np.reshape(grid,(9,9))
                    print(grid)
                else:
                    raise KeyError("Length of Grid (9,9)! = ({},{})".format(h,w))

            else:
                raise KeyError("Type of Grid numpy.ndarray or list != {}".format(type(grid)))
    
    def possible_numbers(self,row,col,num):
        grid = self.grid
        for i in range(9):
            if grid[i][col] == num:
                return False
        for j in range(9):
            if grid[row][j] == num:
                return False

        row_ = (row//3)*3
        col_ = (col//3)*3
        for i in range(3):
            for j in range(3):
                if grid[row_ + i][col_ + j] == num:
                    return False

        return True
    
    def solve_suduko(self):
        grid = self.grid
        for i in range(9):
            for j in range(9):
                if grid[i][j] == -1:
                    for num in range(1,10):
                        if self.possible_numbers(i,j,num):
                            grid[i][j] = num
                            self.solve_suduko()
                            grid[i][j] = -1
                    
                    return
        self.solved_grid = np.matrix(grid)