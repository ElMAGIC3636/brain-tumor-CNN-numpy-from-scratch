import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from relu import Relu 
from Softmax import Softmax
from utils import cross_entropy, cross_entropy_grad
from Adam import Adam
from Maxpooling import MaxPooling
from Flatten import Flatten
from conv import Conv


class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        return x

    def backprop(self, grad):
        return grad * self.mask


# Data Loader #

def load_images_from_folder(folder, img_size=(64,64)):
    images = []
    labels = []
    classes = sorted(os.listdir(folder))
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            img = Image.open(os.path.join(class_path, filename)).convert('L')
            img = img.resize(img_size)
            img = np.array(img) / 255.0
            images.append(img)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels).reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False)
    labels = enc.fit_transform(labels)
    return images, labels, classes

# training # 

if __name__ == "__main__":
    X_train, y_train, classes = load_images_from_folder(r"dataset\training")
    X_test, y_test, _ = load_images_from_folder(r"dataset\Testing")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    conv1 = Conv(32, filter_size=3)
    relu1 = Relu()
    pool1 = MaxPooling()

    conv2 = Conv(64, filter_size=3)
    relu2 = Relu()
    pool2 = MaxPooling()

    conv3 = Conv(128, filter_size=3)
    relu3 = Relu()
    pool3 = MaxPooling()



    flatten = Flatten()
    dropout = Dropout(0.5)
    softmax = Softmax(6*6*128, len(classes))  # correct after 3 pools



    opt_conv1 = Adam(lr=0.002)
    opt_conv2 = Adam(lr=0.002)
    opt_conv3 = Adam(lr=0.002)
    opt_softmax_w = Adam(lr=0.002)
    opt_softmax_b = Adam(lr=0.002)

    batch_size = 32
    epochs = 10
    best_val_loss = float('inf')
    patience, patience_counter = 8, 0

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        total_loss = 0
        total_correct = 0

        for i in range(0, len(X_train), batch_size):
            xb = X_train_shuffled[i:i+batch_size]
            yb = y_train_shuffled[i:i+batch_size]

            out = conv1.forward(xb)
            out = relu1.forward(out)
            out = pool1.forward(out)
            out = conv2.forward(out)
            out = relu2.forward(out)
            out = pool2.forward(out)
            out = conv3.forward(out)
            out = relu3.forward(out)
            out = pool3.forward(out)
            out = flatten.forward(out)
            out = dropout.forward(out, training=True)
            out = softmax.forward(out)

            total_loss += cross_entropy(out, yb)
            total_correct += np.sum(np.argmax(out, axis=1) == np.argmax(yb, axis=1))

            grad = cross_entropy_grad(out, yb)
            grad = softmax.backprop(grad, opt_softmax_w, opt_softmax_b)
            grad = dropout.backprop(grad)
            grad = flatten.backprop(grad)
            grad = pool3.backprop(grad)
            grad = relu3.backprop(grad)
            grad = conv3.backprop(grad, opt_conv3)
            grad = pool2.backprop(grad)
            grad = relu2.backprop(grad)
            grad = conv2.backprop(grad, opt_conv2)
            grad = pool1.backprop(grad)
            grad = relu1.backprop(grad)
            conv1.backprop(grad, opt_conv1)

        val_out = conv1.forward(X_val)
        val_out = relu1.forward(val_out)
        val_out = pool1.forward(val_out)
        val_out = conv2.forward(val_out)
        val_out = relu2.forward(val_out)
        val_out = pool2.forward(val_out)
        val_out = conv3.forward(val_out)
        val_out = relu3.forward(val_out)
        val_out = pool3.forward(val_out)
        val_out = flatten.forward(val_out)
        val_out = dropout.forward(val_out, training=False)
        val_out = softmax.forward(val_out)

        val_loss = cross_entropy(val_out, y_val)
        val_acc = np.mean(np.argmax(val_out, axis=1) == np.argmax(y_val, axis=1)) * 100

        epoch_loss = total_loss / (len(X_train) / batch_size)
        epoch_acc = total_correct / len(X_train) * 100

        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.3f} - Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.3f} - Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        if (epoch+1) % 5 == 0:
            opt_conv1.lr *= 0.8
            opt_conv2.lr *= 0.8
            opt_conv3.lr *= 0.8
            opt_softmax_w.lr *= 0.8
            opt_softmax_b.lr *= 0.8

    test_correct = 0
    for i in range(len(X_test)):
        out = conv1.forward(X_test[i:i+1])
        out = relu1.forward(out)
        out = pool1.forward(out)
        out = conv2.forward(out)
        out = relu2.forward(out)
        out = pool2.forward(out)
        out = conv3.forward(out)
        out = relu3.forward(out)
        out = pool3.forward(out)
        out = flatten.forward(out)
        out = dropout.forward(out, training=False)
        out = softmax.forward(out)

        if np.argmax(out) == np.argmax(y_test[i]):
            test_correct += 1

    test_acc = test_correct / len(X_test) * 100
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
