import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model import model, image_datasets
from train import dataloaders

# Load the trained model
model.load_state_dict(torch.load('model_best.pth'))  # Load the saved model
model.eval()  # Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluate on the test dataset
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report and confusion matrix
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=image_datasets["test"].classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Visualization function
def visualize_predictions(inputs, labels, preds, class_names):
    inputs = inputs.cpu().permute(0, 2, 3, 1).numpy()  # Convert to numpy format
    inputs = (inputs * 0.5 + 0.5)  # Unnormalize
    fig, axes = plt.subplots(1, len(inputs), figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.imshow(inputs[i])
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        ax.axis("off")
    plt.show()

# Visualize predictions
inputs, labels = next(iter(dataloaders["test"]))
inputs, labels = inputs.to(device), labels.to(device)
with torch.no_grad():
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

visualize_predictions(inputs, labels.cpu().numpy(), preds.cpu().numpy(), image_datasets["test"].classes)
