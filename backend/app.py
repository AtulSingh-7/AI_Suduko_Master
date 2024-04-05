import cv2
from function import solved_board
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# implement cors
from flask_cors import CORS
import os

# create a flask app


app = Flask(__name__)
CORS(app)



@app.route('/api/v1/id', methods=['POST'])
def solve_sudoku():
    try:
        
        if 'image' in request.files:
            image_file = request.files['image']
            # print(image_file.filename)
            filename = secure_filename(image_file.filename)
            filepath = f'./uploads/{filename}'
            image_file.save(filepath)

            if os.path.isfile(filepath):
                print('File saved successfully')
            else:
                print('Error saving file')
            img = cv2.imread(filepath)

            solved_grid = solved_board(img) 
            return jsonify({'message': 'Sudoku solved successfully', 'solvedGrid': solved_grid})
        else:
            return jsonify({'error': 'No image file posted'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    # run it on a port
    app.run(port=5000)
