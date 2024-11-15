# array operations
import numpy as np
# read and preprocessing images
import os
import cv2
# save and load model
import pickle
# shuffle data
from sklearn.utils import shuffle
# FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# threading
import threading

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=100, batch_size=32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_mapping = {}

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
            2.0 / (self.input_size + self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(
            2.0 / (self.hidden_size + self.output_size))
        self.b2 = np.zeros((1, self.output_size))

    # Helper method to preprocess an image
    def preprocess_image(self, image):
        # Handle different input types
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image is None:
            raise ValueError("Could not load or process the image")

        # Ensure letter is black on white background
        if np.mean(image) < 127:
            image = 255 - image

        # Binarize the image (Otsu's method)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find bounding box of content
        coords = cv2.findNonZero(binary)
        if coords is None:
            # If no content found, return empty image
            return np.zeros((128, 64), dtype=np.float32)

        x, y, w, h = cv2.boundingRect(coords)

        # Extract the letter
        letter = binary[y:y + h, x:x + w]

        # Get the maximum dimension
        maximum = max(w, h)

        # Create a square white image slightly larger than our letter
        square_size = int(maximum * 1.2)  # 20% padding
        square_img = np.zeros((square_size, square_size), dtype=np.uint8)

        # Calculate center offset
        x_offset = (square_size - w) // 2
        y_offset = (square_size - h) // 2

        # Place the letter in the center of the square image
        square_img[y_offset:y_offset + h, x_offset:x_offset + w] = letter

        # Resize to target size (64x128) using aspect ratio of 1:2
        target_size = (64, 128)
        processed_image = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] range
        processed_image = processed_image.astype(np.float32) / 255.0

        return processed_image

    # Extract HOG features with OpenCV
    def extract_features(self, image):

        # Preprocess the image
        processed_image = self.preprocess_image(image)

        # Convert back to uint8 for HOG
        processed_image = (processed_image * 255).astype(np.uint8)

        # Configure HOG parameters
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        # Create and configure HOG descriptor
        hog = cv2.HOGDescriptor(
            winSize,
            blockSize,
            blockStride,
            cellSize,
            nbins
        )

        # Compute HOG features
        hog_features = hog.compute(processed_image)

        if hog_features is None:
            raise ValueError("Could not compute HOG features")

        return hog_features.flatten()

    # Helper method to visualize preprocessing steps.
    def debug_preprocessing(self, image):
        # Store original image
        debug_images = {'original': image.copy()}

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_images['grayscale'] = image.copy()

        # Binarize
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        debug_images['binary'] = binary.copy()

        # Get bounding box
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)
        bbox_image = image.copy()
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        debug_images['bounding_box'] = bbox_image

        # Get final processed image
        processed = self.preprocess_image(image)
        debug_images['final'] = (processed * 255).astype(np.uint8)

        return debug_images

    # Hyperbolic tangent activation function (-1 to +1)
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2

    # Softmax activation function (for output layer)
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    # Backpropagation algorithm
    def backward(self, X, y, output):
        m = X.shape[0]
        delta2 = output - y
        delta1 = np.dot(delta2, self.W2.T) * self.tanh_derivative(self.z1)
        self.W2 -= self.learning_rate * np.dot(self.a1.T, delta2) / m
        self.b2 -= self.learning_rate * np.sum(delta2, axis=0, keepdims=True) / m
        self.W1 -= self.learning_rate * np.dot(X.T, delta1) / m
        self.b1 -= self.learning_rate * np.sum(delta1, axis=0, keepdims=True) / m

    def train(self, X, y):
        if y.ndim == 1:
            num_classes = self.output_size
            y_one_hot = np.zeros((y.size, num_classes))
            y_one_hot[np.arange(y.size), y] = 1
            y = y_one_hot

        for epoch in range(self.epochs):
            total_error = 0
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output)
                error = -np.mean(np.sum(batch_y * np.log(output + 1e-15), axis=1))
                total_error += error

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, error: {total_error / len(X):.4f}")

    def predict(self, X):
        output = self.forward(X)
        predicted_class = np.argmax(output, axis=1)
        confidence = np.max(output, axis=1)
        return predicted_class, confidence

    def load_and_preprocess_images(self, folder_path):
        X, y = [], []
        label_mapping = {}
        current_label = 0

        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist")
            return X, y

        for letter_folder in os.listdir(folder_path):
            letter_path = os.path.join(folder_path, letter_folder)

            if os.path.isdir(letter_path):
                label_mapping[current_label] = letter_folder
                print(f"Processing letter: {letter_folder}")

                for filename in os.listdir(letter_path):
                    file_path = os.path.join(letter_path, filename)

                    if filename.endswith(".png"):
                        image = cv2.imread(file_path)
                        if image is None:
                            print(f"Warning: Couldn't read image {file_path}")
                            continue

                        features = self.extract_features(image)
                        X.append(features)
                        y.append(current_label)

                current_label += 1

        self.label_mapping = label_mapping
        print(f"Total images processed: {len(X)}")
        X, y = shuffle(np.array(X), np.array(y), random_state=42)
        return X, y

    def split_data(self, X, y, test_size=0.2):
        train_size = int(len(X) * (1 - test_size))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test):
        predictions, _ = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, filename='greek_letters_model.pkl'):
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'label_mapping': self.label_mapping
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved as {filename}")

    def load_model(self, filename='greek_letters_model.pkl'):
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.W1 = model_data['W1']
            self.b1 = model_data['b1']
            self.W2 = model_data['W2']
            self.b2 = model_data['b2']
            self.label_mapping = model_data['label_mapping']
            print("Model loaded successfully")
        except FileNotFoundError:
            print(f"Error: Model file '{filename}' not found.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")


# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize neural network
input_size = 3780  # HOG feature size for 64x128 image
hidden_size = 392
output_size = 24
network = NeuralNetwork(input_size, hidden_size, output_size)

try:
    network.load_model('./greek_letters_model.pkl')
except:
    print("No model found, will need to train first")


@app.get('/')
async def health_check():
    return {"message": "Ok!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        features = network.extract_features(contents)
        predicted_class, confidence = network.predict(features.reshape(1, -1))
        predicted_letter = network.label_mapping.get(predicted_class[0], "Unknown")

        return {
            "predicted_letter": predicted_letter,
            "confidence": float(confidence[0]),
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def run_api():
    uvicorn.run(app, host="localhost", port=8000)

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1. Training Mode")
        print("2. Testing Mode")
        print("3. Run API")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            training_menu(network)
        elif choice == "2":
            testing_menu(network)
        elif choice == "3":
            api_thread = threading.Thread(target=run_api)
            api_thread.start()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option, please try again")

def training_menu(network):
    X = None
    y = None
    while True:
        print("\nTraining Menu:")
        print("1. Load and preprocess images")
        print("2. Shuffle and split data")
        print("3. Train model")
        print("4. Evaluate model")
        print("5. Save model")
        print("0. Back to main menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            folder_path = input("Enter the path to 'Greek_Letters' folder: ")
            if os.path.exists(folder_path):
                X, y = network.load_and_preprocess_images(folder_path)
            else:
                print("Error: Folder does not exist")

        elif choice == "2":
            if X is not None and y is not None:
                print("Shuffling and splitting data...")
                X_train, X_test, y_train, y_test = network.split_data(X, y)
                print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "3":
            if X is not None and y is not None:
                if 'X_train' not in locals():
                    print("Error: Data is not split yet. Please shuffle and split data first (option 2).")
                    continue
                print("Training the model...")
                network.train(X_train, y_train)
                print("Training completed!")
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "4":
            if X is not None and y is not None:
                if 'X_test' not in locals():
                    print("Error: Data is not split yet. Please shuffle and split data first (option 2).")
                    continue
                print("Evaluating the model...")
                network.evaluate(X_test, y_test)
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "5":
            print("Saving the model...")
            network.save_model()
            print("Model saved successfully!")

        elif choice == "0":
            break

        else:
            print("Invalid option, please try again")

def testing_menu(network):
    model_loaded = False
    while True:
        print("\nTesting Menu:")
        print("1. Load model")
        print("2. Test model with image")
        print("0. Back to main menu")
        choice = input("\nEnter your choice: ")

        if choice == "1":
            filename = input("Enter the filename of the model to load (default: greek_letters_model.pkl): ").strip()
            if not filename:
                filename = 'greek_letters_model.pkl'
            try:
                network.load_model(filename)
                model_loaded = True
            except Exception as e:
                print(f"Error loading model: {str(e)}")

        elif choice == "2":
            if not model_loaded:
                print("Error: No model loaded. Please load a model first (Option 1).")
            else:
                test_image_path = input("Enter the path to the image for testing: ")
                try:
                    # Extract features directly using the new preprocessing pipeline
                    features = network.extract_features(test_image_path)
                    # Make prediction
                    predicted_class, confidence = network.predict(features.reshape(1, -1))
                    predicted_letter = network.label_mapping.get(predicted_class[0], "Unknown")
                    print(f"Predicted letter: {predicted_letter}")
                    print(f"Confidence: {confidence[0]:.2f}")
                    # Load and display the original image
                    original_image = cv2.imread(test_image_path)
                    if original_image is not None:
                        cv2.imshow("Original Image", original_image)
                        # Show the preprocessed image
                        processed_image = network.preprocess_image(test_image_path)
                        processed_image = (processed_image * 255).astype(np.uint8)
                        cv2.imshow("Preprocessed Image", processed_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                except Exception as e:
                    print(f"Error processing image: {str(e)}")

        elif choice == "0":
            break
        else:
            print("Invalid option, please try again")

if __name__ == "__main__":
    main_menu()