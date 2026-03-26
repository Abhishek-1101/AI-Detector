AI vs Real Image Detector
This project is a Deep Learning-based system designed to classify images as either Real or AI-Generated. It utilizes a Convolutional Neural Network (CNN) trained on the CIFAKE dataset to identify subtle patterns and artifacts typical of synthetic imagery.
📁 Project Structure
dataset/: Contains the training and testing image sets (Real vs AI).

model/: Contains the trained .h5 model file.

app.py: A Streamlit-based web application for easy image testing.

train_model.py: The script used to train the neural network.
🚀 Key FeaturesHigh Accuracy: Optimized for the CIFAKE dataset.User Interface: Simple drag-and-drop image testing via Streamlit.Preprocessing: Automated image resizing and normalization ($128 \times 128$ px).
🛠️ How to Run
Install dependencies: pip install tensorflow streamlit opencv-python pillow

Run the application: streamlit run app.py
