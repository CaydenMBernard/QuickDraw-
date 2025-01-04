# QuickDraw
This project features a Feedforward Neural Network (FNN) built from scratch using calculus, linear algebra, and NumPy, trained on the Google Quick, Draw! dataset. The neural network is capable of recognizing ten different types of doodles with impressive accuracy.

To demonstrate the model's functionality, a Pygame-based drawing application was developed, allowing users to draw on a canvas and receive real-time predictions from the FNN. The application continuously interacts with the neural network, providing immediate feedback on the type of doodle being drawn.
# Key Features
- Custom Neural Network Implementation:
    - Built entirely from scratch, without external machine learning frameworks.
      Supports forward propagation, backpropagation, and training with gradient descent.
- Training on Google Quick, Draw! Dataset:
    - Recognizes 10 distinct categories of doodles.
      Dataset preprocessed and scaled for optimal training performance.
- Real-Time Drawing Application:
    - Developed with Pygame, offering a simple and interactive drawing interface.
      Constantly queries the FNN to provide real-time predictions for user drawings.
# Imports Used
Python: Core programming language for the project.
NumPy: Numerical computations and matrix operations.
Pygame: Interactive drawing application.
Pillow: Image processing for doodle preprocessing
