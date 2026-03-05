# PyTorch Digit Classifier (MNIST)

A Deep Learning based **Handwritten Digit Classifier** built using **PyTorch** and deployed as an interactive **Streamlit Web App**.

Users can **draw a digit on canvas or upload an image**, and the AI model predicts the digit (0–9).

---

## 🚀 Live Demo

Try the deployed web app here:

https://pytorch-digitproject-8jxe8c3vhpz878msvjoz5s.streamlit.app/

Draw a digit or upload an image and the model will predict the number.

---

## 🧠 Model Details

Dataset: **MNIST Handwritten Digits**

Number of classes: **10**

Digits:

```
0 1 2 3 4 5 6 7 8 9
```

Framework: **PyTorch**

Architecture:

```
Input (28x28)
↓
Flatten
↓
Dense (128)
↓
Dense (64)
↓
Dense (10)
↓
Softmax
```

---

## ✨ Features

* Draw digit using canvas
* Upload digit image
* Automatic image preprocessing
* Digit prediction (0–9)
* Confidence score
* Top-3 predictions
* Processed image preview
* Interactive Streamlit UI
* Live cloud deployment

---

## 🏗️ Project Workflow

```
User Input (Canvas / Upload Image)
        ↓
Image Preprocessing (Grayscale + Resize 28x28)
        ↓
Convert to Tensor
        ↓
PyTorch Neural Network Model
        ↓
Prediction (0–9)
        ↓
Confidence & Top Predictions
```

---

## 📂 Project Structure

```
pytorch-digitproject
│
├── model
│   └── mnist_model.pth
│
├── app.py
├── train.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/pytorch-digitproject.git
```

Go to project folder:

```
cd pytorch-digitproject
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Run the Streamlit app:

```
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 🧪 Train the Model

To train the PyTorch model run:

```
python train.py
```

The trained model will be saved as:

```
model/mnist_model.pth
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pillow
* Streamlit
* streamlit-drawable-canvas

---

## 📸 Demo

Example output:

```
Draw Digit
↓
Prediction: 7
Confidence: 98%
Top Predictions:
7 → 98%
1 → 1%
9 → 1%
```

---

## 📌 Future Improvements

* CNN architecture for better accuracy
* Mobile friendly UI
* Real-time drawing prediction
* Model optimization
* Docker deployment

---

## 👨‍💻 Author

Deep Learning project built using **PyTorch and Streamlit** for handwritten digit recognition.
