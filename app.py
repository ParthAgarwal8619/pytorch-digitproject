import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------
# MODEL
# --------------------

class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):

        x = x.view(-1,28*28)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x


model = Net()
model.load_state_dict(torch.load("model/mnist_model.pth",map_location="cpu"))
model.eval()

# --------------------
# STREAMLIT UI
# --------------------

st.title("Draw Digit Classifier")

st.write("Draw a digit (0-9) below")

# canvas
canvas_result = st_canvas(

    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --------------------
# PREDICTION
# --------------------

if canvas_result.image_data is not None:

    img = canvas_result.image_data

    img = Image.fromarray(img.astype("uint8")).convert("L")

    img = img.resize((28,28))

    img_array = np.array(img)

    img_array = img_array/255.0

    st.subheader("Processed Image")

    st.image(img_array,width=150)

    img_array = img_array.reshape(1,28*28)

    img_tensor = torch.tensor(img_array).float()

    output = model(img_tensor)

    probabilities = torch.softmax(output,dim=1)

    pred = torch.argmax(probabilities,1).item()

    confidence = probabilities[0][pred].item()*100

    st.subheader("Prediction")

    st.write("Digit:",pred)

    st.write("Confidence:",round(confidence,2),"%")

    # Top 3
    st.subheader("Top 3 Predictions")

    probs = probabilities.detach().numpy()[0]

    top3 = probs.argsort()[-3:][::-1]

    for i in top3:

        st.write("Digit",i,"→",round(probs[i]*100,2),"%")