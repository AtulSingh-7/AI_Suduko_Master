import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../model(2).h5')

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocessing(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  img = img/255
  return img

def process_image(img):
 img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
 img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
 img_thresh = cv2.bitwise_not(img_thresh)
 contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 contours = sorted(contours, key=cv2.contourArea, reverse=True)
 contour = contours[0]
 peri = cv2.arcLength(contour, True)
 approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
 grid = four_point_transform(img, approx.reshape(4, 2))
 img=grid
 grid_size = (9, 9)
 cell_size = (img.shape[0]//grid_size[0], img.shape[1]//grid_size[1])
 cells = [[None]*grid_size[1] for _ in range(grid_size[0])]
 for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        cell = img[(i*cell_size[0]):((i+1)*cell_size[0]), (j*cell_size[1]):((j+1)*cell_size[1])]
        cells[i][j] = cell
 new_cells = [[None]*grid_size[1] for _ in range(grid_size[0])]

 for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        # cv2_imshow(cells[i][j][10:-10, 10:-10])
        new_cells[i][j] = cells[i][j][10:-10, 10:-10]
  
 return new_cells

def predict_cells(new_cells, model):
  sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]
  score_grid = [[0 for _ in range(9)] for _ in range(9)]
  for i in range(9):
    for j in range(9):

        cell = new_cells[i][j]
        img=np.asarray(cell)
        img=cv2.resize(img,(32,32))
        # img=cv2.resize(img,(28,28))
        img=preprocessing(img)
        img=img.reshape(1,32,32,1)
        # img=img.reshape(1,28,28,1)
        classid=model.predict(img)

        if np.max(classid) > 0.80:
         score_grid[i][j] = round(float(classid.max()),2)
         sudoku_grid[i][j] = int(classid.argmax(axis=1))

  return sudoku_grid, score_grid

def is_valid(board, row, col, num):
    # Check the number in the row
    for x in range(9):
        if board[row][x] == num:
            return False

    # Check the number in the column
    for x in range(9):
        if board[x][col] == num:
            return False

    # Check the number in the 3x3 matrix
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False
    return True

def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_sudoku(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def solved_board(img):
    new_cells=process_image(img)
    sudoku_grid=[[0 for _ in range(9)] for _ in range(9)]
    sudoku_grid= predict_cells(new_cells, model)[0]
    solve_sudoku(sudoku_grid)
    return sudoku_grid

