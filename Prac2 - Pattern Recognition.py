from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report


######################### Data Preparation #########################
def data_prep_basic():
    # Download the data, if not already on disk and load it as numpy arrays
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    # Split into a training set and a test set using a stratified k fold
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150

    # Center data
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean
    return n_components, h, w, X_train, X_test, y_train, y_test, n_classes, n_samples, n_features, target_names

def data_prep_CNN():
    # Download the data, if not already on disk and load it as numpy arrays
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    X = lfw_people.images
    Y = lfw_people.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    X_train = X_train[:, np.newaxis, :, :] # Only 1 channel
    X_test = X_test[:, np.newaxis, :, :]
    return X_test, X_train, y_train, y_test


n_components, h, w, X_train, X_test, y_train, y_test, n_classes, n_samples, n_features, target_names = data_prep_basic()
X_test2, X_train2, y_train2, y_test2 = data_prep_CNN()

def PCA_eigendecomposition(X_train, X_test, n_components, h, w):
    # Eigen-decomposition
    # Singular Value Decomposition
    U, S, V = np.linalg.svd(X_train, full_matrices=False)
    svd_out = U, S, V
    components = V[:n_components]
    eigenfaces = components.reshape((n_components, h, w))

    #project into PCA subspace
    X_transformed = np.dot(X_train, components.T)
    print(X_transformed.shape)
    X_test_transformed = np.dot(X_test, components.T)
    print(X_test_transformed.shape)
    
    return eigenfaces, X_transformed, X_test_transformed, svd_out

def PCA_torch(X_train, X_test, n_components, h, w):
    # Convert to pytorch tensor
    X_train_tensor = torch.as_tensor(X_train)
    X_test_tensor = torch.as_tensor(X_test)
    # Flatten to 2D
    X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
    X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), -1)
    # Singular Value Decomposition
    U, S, V = torch.pca_lowrank(X_train_tensor, q=n_components, center=False)
    svd_out = U, S, V
    # Could also use matmul()
    components = V.T[:n_components]
    eigenfaces = components.reshape((n_components, h, w))
    
    #project into PCA subspace (apply dot product)
    X_transformed = torch.mm(X_train_tensor, components.T)
    print(X_transformed.shape)
    X_test_transformed = torch.mm(X_test_tensor, components.T)
    print(X_test_transformed.shape)
    
    return eigenfaces, X_transformed, X_test_transformed, svd_out

# eigenfaces, X_transformed, X_test_transformed, svd_out = PCA_torch(X_train2, X_test2, n_components,h ,w)

######################### Data Analysis #########################

# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, eigenface_titles, h, w)


# Dimensionality reduction compactness plot
def plot_dimensionality_reduction_compactness(n_samples, n_components, svd_out):
    explained_variance = (svd_out[1] ** 2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    ratio_cumsum = np.cumsum(explained_variance_ratio)
    print(ratio_cumsum.shape)
    eigenvalueCount = np.arange(n_components)
    plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
    plt.title('Compactness')
    plt.show()

# plot_dimensionality_reduction_compactness(n_samples, n_components, svd_out)

# Build random forest
def RFC(X_transformed, X_test_transformed, y_train, y_test, verbose=True):
    estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
    estimator.fit(X_transformed, y_train) #expects X as [n_samples, n_features]
    predictions = estimator.predict(X_test_transformed)
    correct = predictions==y_test
    total_test = len(X_test_transformed)
    if verbose:
        print("Gnd Truth:", y_test)
        print("Total Testing", total_test)
        print("Predictions", predictions)
        print("Which Correct:",correct)
        print("Total Correct:",np.sum(correct))
        print("Accuracy:",np.sum(correct)/total_test)
        print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))
    return classification_report(y_test, predictions, target_names=target_names, zero_division=0)

# RFC_report = RFC(X_transformed, X_test_transformed, y_train, y_test, verbose=True)

class CNN_Model(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, 
                 conv_filters=32, kernel_size=3, 
                 dense_units=128, input_height=50, input_width=37):
        super(CNN_Model, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=conv_filters, 
                               kernel_size=kernel_size, 
                               padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=conv_filters, 
                               out_channels=conv_filters, 
                               kernel_size=kernel_size, 
                               padding=1)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute output size after all layers
        def compute_output_size(size, kernel_size, stride, padding):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        # Dimensions after first convolution
        height_after_conv1 = compute_output_size(input_height, kernel_size, 1, 1)
        width_after_conv1 = compute_output_size(input_width, kernel_size, 1, 1)
        
        # Dimensions after first pooling
        height_after_pool1 = compute_output_size(height_after_conv1, 2, 2, 0)
        width_after_pool1 = compute_output_size(width_after_conv1, 2, 2, 0)
        
        # Dimensions after second convolution
        height_after_conv2 = compute_output_size(height_after_pool1, kernel_size, 1, 1)
        width_after_conv2 = compute_output_size(width_after_pool1, kernel_size, 1, 1)
        
        # Dimensions after second pooling
        height_after_pool2 = compute_output_size(height_after_conv2, 2, 2, 0)
        width_after_pool2 = compute_output_size(width_after_conv2, 2, 2, 0)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(conv_filters * height_after_pool2 * width_after_pool2, dense_units)
        self.fc2 = nn.Linear(dense_units, num_classes)
        
        # Activation Function
        self.relu = nn.ReLU()
        
        # Softmax for output layer
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)  # Flatten the tensor for the dense layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def CNN(X_train, X_test, y_train, y_test, 
        input_channels=1, num_classes=n_classes, 
        conv_filters=32, kernel_size=3, 
        dense_units=128, 
        learning_rate=0.0013, num_epochs=20, batch_size=64, target_names=None, verbose=True):
    
    
    
    # Convert input data to PyTorch tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    height, width = X_train.shape[2:4]
    
    # Instantiate the model, loss function, and optimizer
    model = CNN_Model(input_channels=input_channels, 
                      num_classes=num_classes, 
                      conv_filters=conv_filters, 
                      kernel_size=kernel_size, 
                      dense_units=dense_units,
                      input_height=height,
                      input_width=width)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        _, predicted_labels = torch.max(predictions, 1)
        
        correct = (predicted_labels == y_test).numpy()
        total_test = len(X_test)
        if verbose:
            print("Total Testing:", total_test)
            print("Predictions:", predicted_labels.numpy())
            print("Which Correct:", correct)
            print("Total Correct:", np.sum(correct))
            print("Accuracy:", np.sum(correct) / total_test)
            print(classification_report(y_test.numpy(), predicted_labels.numpy(), target_names=target_names, zero_division=0))
    
    return classification_report(y_test.numpy(), predicted_labels.numpy(), target_names=target_names, zero_division=0)


CNN_report = CNN(X_train2, X_test2, y_train2, y_test2, verbose=False) # No PCA
print(CNN_report)


