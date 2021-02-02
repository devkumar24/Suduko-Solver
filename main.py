import numpy as np
import cv2
import suduko_solver
import extract_no
from processing_images.process_image import *
import time

image_path = "./images/before/sudoku-puzzle-863979.jpg"
model_path = "./model/best_model/digit_classifier1.h5"

def get_sol(image_path : str = "",model_path : str = "",showGrid = True):
    digit = pre_process_images(image_path)
    grid = extract_no.extract_number(digit,model_path)
    if showGrid:
        displayBoard(grid)
#     grid = extract_no.save_grid(grid,filename="./puzzles.txt")
    suduko = suduko_solver.Suduko(filename="./puzzles.txt")
    suduko.solve_suduko()
    solve_grid = suduko.solved_grid
    
    return grid,solve_grid

def displayBoard(sudoku):
    for i in range(9):
        if i % 3 == 0:
            if i == 0:
                print(" ┎─────────┰─────────┰─────────┒")
            else:
                print(" ┠─────────╂─────────╂─────────┨")

        for j in range(9):
            if j % 3 == 0:
                print(" ┃ ", end=" ")

            if j == 8:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", " ┃")
            else:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", end=" ")

    print(" ┖─────────┸─────────┸─────────┚")
    

if __name__ == '__main__':
    try:
        start = time.time()

        grid,solved_grid = get_sol(image_path,model_path)
        solved_grid = np.array(solved_grid)
        end = time.time()
        
        if -1 in solved_grid:
            print("""Suduko Puzzle has can't be solved due to some unwanted values present in GRID.""")
        else:
            displayBoard(solved_grid)
        print(int(end - start),"Seconds")
        
    except:
        fmt = 'usage: {} image_path'
        print(fmt.format(image_path.split(sep="/")[-1]))
        print('[ERROR]: Image not found')