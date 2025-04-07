import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_circles, make_classification, make_swiss_roll
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio

# Initialize the asyncio event loop to avoid errors
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit page configuration
st.set_page_config(page_title="Visualizing MLP Transformations", layout="wide")

# Application title and description
st.title("Visualizing MLP Transformations")
st.markdown("""
This application allows you to visualize how a neural network (MLP) transforms the data space across its layers, 
illustrating the "Manifold Hypothesis" which suggests that real-world data lies on lower-dimensional manifolds within the input space.
""")

# MLP class with hooks to capture activations
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.pre_activations = []
        self.post_activations = []
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
    def forward(self, x):
        self.pre_activations = []
        self.post_activations = []
        
        # Save the input as the initial activation
        self.pre_activations.append(x)
        self.post_activations.append(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.pre_activations.append(x)
            
            if i < len(self.layers) - 1:
                x = self.activation(x)
            else:
                # For multi-class classification or binary classification
                x = torch.softmax(x, dim=1) if x.shape[1] > 1 else torch.sigmoid(x)
            
            self.post_activations.append(x)
            
        return x
    
    def get_activations(self):
        return self.pre_activations, self.post_activations

# Function to generate datasets
def generate_dataset(dataset_name, n_samples=1000):
    if dataset_name == "Two Circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif dataset_name == "Two Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_name == "Linear Classification":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=42, n_clusters_per_class=1)
    elif dataset_name == "Swiss Roll":
        X, y = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
        X = X[:, [0, 2]]  # Keep only 2 dimensions for visualization
    elif dataset_name == "Blobs":
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=4, n_features=2, random_state=42)
    elif dataset_name == "XOR":
        X = np.random.uniform(-1, 1, (n_samples, 2))
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized.")
    
    return X, y

# Function to plot training data
def plot_dataset(X, y):
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str),
                     labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'color': 'Class'},
                     title="Training Data")
    return fig

# Function to apply PCA if necessary
def apply_pca_if_needed(data, n_components=2):
    if data.shape[1] < n_components:
        second_dim = np.zeros((data.shape[0], n_components - data.shape[1]))
        data = np.hstack((data, second_dim))
        return data, None
    if data.shape[1] == n_components:
        return data, None
    pca = PCA(n_components=min(n_components, data.shape[1]))
    data_reduced = pca.fit_transform(data)
    if data_reduced.shape[1] < n_components:
        second_dim = np.zeros((data_reduced.shape[0], n_components - data_reduced.shape[1]))
        data_reduced = np.hstack((data_reduced, second_dim))
    return data_reduced, pca

# Function to visualize each layer with two side-by-side plots: left is pre-activation and right is post-activation
def visualize_all_layers_side_by_side(model, X, y):
    pre_acts, post_acts = model.get_activations()
    num_layers = len(pre_acts)
    
    # Define a fixed color palette for all layers using Plotly's qualitative palette
    unique_classes = np.unique(y)
    colors = px.colors.qualitative.Plotly
    class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
    
    # Create subplot titles for each layer (pre and post activation)
    subplot_titles = [f"Layer {i} - Pre Activation" if j == 0 else f"Layer {i} - Post Activation"
                      for i in range(num_layers) for j in range(2)]
    
    fig = make_subplots(
        rows=num_layers, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    for i in range(num_layers):
        pre_i = pre_acts[i].detach().numpy()
        post_i = post_acts[i].detach().numpy()
        pre_i_reduced, _ = apply_pca_if_needed(pre_i)
        post_i_reduced, _ = apply_pca_if_needed(post_i)
        
        # Show legend only for the first row
        showlegend_flag = True if i == 0 else False
        
        for class_label in unique_classes:
            mask = (y == class_label)
            # Column 1: Pre Activation with fixed color
            fig.add_trace(
                go.Scatter(
                    x=pre_i_reduced[mask, 0],
                    y=pre_i_reduced[mask, 1],
                    mode='markers',
                    name=f'Class {class_label}',
                    marker=dict(symbol='circle', size=8, opacity=0.7, color=class_colors[class_label]),
                    showlegend=showlegend_flag
                ),
                row=i+1, col=1
            )
            # Column 2: Post Activation with the same fixed color
            fig.add_trace(
                go.Scatter(
                    x=post_i_reduced[mask, 0],
                    y=post_i_reduced[mask, 1],
                    mode='markers',
                    name=f'Class {class_label}',
                    marker=dict(symbol='square', size=8, opacity=0.7, color=class_colors[class_label]),
                    showlegend=False
                ),
                row=i+1, col=2
            )
        fig.update_xaxes(title_text="Component 1", row=i+1, col=1)
        fig.update_yaxes(title_text="Component 2", row=i+1, col=1)
        fig.update_xaxes(title_text="Component 1", row=i+1, col=2)
        fig.update_yaxes(title_text="Component 2", row=i+1, col=2)
    
    fig.update_layout(height=400*num_layers, title_text="Layer Transformations (Pre and Post Activation)")
    return fig

# Training function
def train_model(model, X, y, epochs, batch_size, learning_rate):
    X_tensor = torch.FloatTensor(X)
    if len(np.unique(y)) <= 2:
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
    else:
        y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss() if len(np.unique(y)) > 2 else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            st.sidebar.write(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, losses

# Initialize session_state for the trained model
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

# Sidebar: Configuration
st.sidebar.header("Configuration")

dataset_options = ["Two Circles", "Two Moons", "Linear Classification", "Swiss Roll", "Blobs", "XOR"]
selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)

n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=2000, value=1000, step=100)
X, y = generate_dataset(selected_dataset, n_samples)

st.subheader("Training Data Visualization")
fig_data = plot_dataset(X, y)
st.plotly_chart(fig_data, use_container_width=True)

st.sidebar.subheader("MLP Configuration")
input_dim = X.shape[1]
output_dim = len(np.unique(y))
if output_dim == 2:
    output_dim = 1

n_hidden_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=5, value=2)
hidden_dims = []
for i in range(n_hidden_layers):
    hidden_dim = st.sidebar.slider(f"Neurons in Hidden Layer {i+1}", min_value=2, max_value=100, value=32)
    hidden_dims.append(hidden_dim)

activation_options = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid(), "LeakyReLU": nn.LeakyReLU()}
selected_activation = st.sidebar.selectbox("Activation Function", list(activation_options.keys()))
activation = activation_options[selected_activation]

if st.session_state["trained_model"] is not None:
    model = st.session_state["trained_model"]
else:
    model = MLP(input_dim, hidden_dims, output_dim, activation)

st.sidebar.subheader("Training Parameters")
epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=200, value=50)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, value=32)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01
)

train_button = st.sidebar.button("Train Model")

# Compute activations with a forward pass
with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    outputs = model(X_tensor)

st.subheader("Layer Transformations (Pre and Post Activation)")
fig_side_by_side = visualize_all_layers_side_by_side(model, X_tensor, y)
st.plotly_chart(fig_side_by_side, use_container_width=True)

if train_button:
    st.sidebar.info("Training in progress...")
    trained_model, losses = train_model(model, X, y, epochs, batch_size, learning_rate)
    st.session_state["trained_model"] = trained_model
    fig_loss = px.line(y=losses, x=list(range(len(losses))),
                       labels={"y": "Loss", "x": "Epoch"},
                       title="Loss During Training")
    st.plotly_chart(fig_loss)
    st.success("Training completed! Visualizations have been updated.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About the Manifold Hypothesis

The "Manifold Hypothesis" is a fundamental concept in deep learning which postulates that natural, high-dimensional data 
lies approximately on a lower-dimensional manifold within the input space.

Neural networks perform a series of transformations to "unfold" these complex manifolds, making the data linearly separable.

This application allows you to visualize this process in real time across the different layers of an MLP.
""")
