import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ------------------ Alzheimer Model ------------------

class AlzheimerCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alz_model = AlzheimerCNN()
alz_model.load_state_dict(torch.load("alzmodel.pth", map_location=device))
alz_model.eval()

# ------------------ Prediction ------------------

recommendation_dict = {
    "Non Demented": "No signs of Alzheimer's detected. Maintain a healthy brain with mental activities and regular checkups.",
    "Demented": "Signs of dementia detected. Consult a neurologist for proper diagnosis and treatment options.",
}

def predict(image):
    img = Image.open(image).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    outputs = alz_model(img_tensor)
    class_names = ["Non Demented", "Demented"]

    probs = torch.softmax(outputs, dim=1)[0]
    pred_label = class_names[torch.argmax(probs).item()]
    recommendation = recommendation_dict.get(pred_label, "No recommendation available.")

    return f"🧠 Prediction: {pred_label}\n\n📌 Recommendation: {recommendation}"

# ------------------ Gradio UI ------------------

with gr.Blocks(theme=gr.themes.Base(), title="Alzheimer Detection") as demo:
    gr.Markdown("## Upload an MRI scan for Alzheimer classification")
    image_input = gr.Image(type="filepath", label="Upload MRI scan")
    output_text = gr.Textbox(label="Prediction and Recommendation")
    submit_btn = gr.Button("Predict")
    submit_btn.click(fn=predict, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
