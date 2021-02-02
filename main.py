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
        print(grid)
    grid = extract_no.save_grid(grid,filename="./puzzles.txt")
    suduko = suduko_solver.Suduko(filename="./puzzles.txt")
    suduko.solve_suduko()
    solve_grid = suduko.solved_grid
    
    return solve_grid

if __name__ == '__main__':
    try:
        start = time.time()

        solved_grid = get_sol(image_path,model_path)
        end = time.time()
        
        if -1 in solved_grid:
            print("""Suduko Puzzle has can't be solved due to some unwanted values present in GRID.""")
        else:
            print(solved_grid)
        print(int(end - start),"Seconds")
        
    except:
        fmt = 'usage: {} image_path'
        print(fmt.format(image_path.split(sep="/")[-1]))
        print('[ERROR]: Image not found')