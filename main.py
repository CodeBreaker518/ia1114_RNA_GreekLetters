import numpy as np
import cv2
import os
import pickle
from sklearn.utils import shuffle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=100, batch_size=32):
        self.input_size = input_size #features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_mapping = {}

        # Weights initialization using He initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / (self.hidden_size + self.output_size))
        self.b2 = np.zeros((1, self.output_size))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        # Output layer error
        delta2 = output - y

        # Hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * self.tanh_derivative(self.z1)

        # Update weights and biases
        self.W2 -= self.learning_rate * np.dot(self.a1.T, delta2)
        self.b2 -= self.learning_rate * np.sum(delta2, axis=0, keepdims=True)
        self.W1 -= self.learning_rate * np.dot(X.T, delta1)
        self.b1 -= self.learning_rate * np.sum(delta1, axis=0, keepdims=True)

    def train(self, X, y):
        # Convert labels to one-hot encoding if they are not already
        if y.ndim == 1:
            num_classes = self.output_size
            y_one_hot = np.zeros((y.size, num_classes))
            y_one_hot[np.arange(y.size), y] = 1
            y = y_one_hot

        for epoch in range(self.epochs):
            total_loss = 0
            # Training in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]

                # Forward propagation
                output = self.forward(batch_X)

                # Backward propagation
                self.backward(batch_X, batch_y, output)

                # Calculate loss
                loss = -np.mean(np.sum(batch_y * np.log(output + 1e-15), axis=1))
                total_loss += loss

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def extract_features(self, image):
        resized_image = cv2.resize(image, (64, 128))  # Tamaño adecuado para HOG
        resized_image = np.array(resized_image, dtype=np.uint8)  # Asegura que sea un array de tipo uint8

        # Verifica si la imagen tiene 1 canal (grayscale)
        if len(resized_image.shape) != 2:
            print(f"Warning: Image is not grayscale! Shape: {resized_image.shape}")
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        hog = cv2.HOGDescriptor()

        # Asegúrate de que la imagen esté en el formato correcto antes de pasarla al HOG
        if resized_image.shape[0] == 128 and resized_image.shape[1] == 64:
            hog_features = hog.compute(resized_image)
        else:
            print(f"Warning: Image has invalid dimensions: {resized_image.shape}")
            hog_features = np.zeros((3780, 1))  # Placeholder para evitar errores

        return hog_features.flatten()

    def save_hog_features_to_csv(self, path, X, y):
        import pandas as pd
        # Convertir los datos en un DataFrame de pandas
        data = pd.DataFrame(X)
        data['label'] = y
        data.to_csv(path, index=False)
        print(f"HOG features saved to {path}")

    def load_and_preprocess_images(self, folder_path):
        X, y = [], []
        label_mapping = {}
        current_label = 0
        input_size_determined = False

        for letter_folder in os.listdir(folder_path):
            letter_path = os.path.join(folder_path, letter_folder)

            if os.path.isdir(letter_path):
                label_mapping[current_label] = letter_folder
                print(f"Processing letter: {letter_folder}")

                for filename in os.listdir(letter_path):
                    file_path = os.path.join(letter_path, filename)

                    if filename.endswith(".png"):
                        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                        if image is None:
                            print(f"Warning: Couldn't read image {file_path}")
                            continue

                        features = self.extract_features(image)  # Extrae características HOG

                        # Solo determina el input_size una vez
                        if not input_size_determined:
                            self.input_size = features.shape[0]
                            self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(
                                2.0 / (self.input_size + self.hidden_size))
                            input_size_determined = True

                        X.append(features)
                        y.append(current_label)

                        # Mostrar solo la primera imagen procesada
                        if current_label == 0 and len(y) == 1:  # Solo mostrar la primera imagen de la primera letra
                            print(f"Showing original and preprocessed image for: {letter_folder}")

                            # Mostrar la imagen original
                            cv2.imshow("Original Image", image)

                            # Preprocesar y mostrar la imagen redimensionada (para HOG)
                            preprocessed_image = cv2.resize(image, (64, 128))  # Tamaño adecuado para HOG
                            cv2.imshow("Preprocessed Image", preprocessed_image)

                            # Esperar y cerrar la ventana de las imágenes
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                current_label += 1

        self.label_mapping = label_mapping
        print(f"Total images processed: {len(X)}")

        # Mezclar los datos (imagenes y etiquetas)
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
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, filename='greek_letters_model.pkl'):
        try:
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
        except FileNotFoundError:
            print(f"Error: File {filename} not found")

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
            print(f"Error: File {filename} not found")

def main_menu():
    input_size = 784  # Valor inicial, será redefinido en `load_and_preprocess_images`
    hidden_size = 392
    output_size = 24

    network = NeuralNetwork(input_size, hidden_size, output_size)

    while True:
        print("\nMain Menu:")
        print("1. Training Mode")
        print("2. Testing Mode")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            training_menu(network)
        elif choice == "2":
            testing_menu(network)
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
        print("2. Extract HOG features and save to CSV")
        print("3. Shuffle and split data")
        print("4. Train model")
        print("5. Evaluate model")
        print("6. Save model")
        print("0. Back to main menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            image_folder = input("Enter the path to 'Greek_Letters' folder: ")
            if os.path.exists(image_folder):
                X, y = network.load_and_preprocess_images(image_folder)
            else:
                print("Error: Folder does not exist")

        elif choice == "2":
            if X is not None and y is not None:
                print("Extracting HOG features...")
                # Aquí podrías guardar las características HOG en un CSV
                features_csv_path = input("Enter path to save HOG features CSV: ")
                network.save_hog_features_to_csv(features_csv_path, X, y)
                print("HOG features extracted and saved to CSV")
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "3":
            if X is not None and y is not None:
                print("Shuffling and splitting data...")
                X_train, X_test, y_train, y_test = network.split_data(X, y)
                print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "4":
            if X is not None and y is not None:
                if 'X_train' not in locals() or 'y_train' not in locals():
                    print("Error: Data is not split yet. Please shuffle and split data first (option 3).")
                    continue
                print("Training the model...")
                network.train(X_train, y_train)
                print("Training completed!")
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "5":
            if X is not None and y is not None:
                if 'X_test' not in locals() or 'y_test' not in locals():
                    print("Error: Data is not split yet. Please shuffle and split data first (option 3).")
                    continue
                print("Evaluating the model...")
                network.evaluate(X_test, y_test)
            else:
                print("Error: You must first load and preprocess the images (option 1)")

        elif choice == "6":
            print("Saving the model...")
            model_filename = input("Enter the filename to save the model (default: greek_letters_model.pkl): ").strip()
            if not model_filename:
                model_filename = 'greek_letters_model.pkl'
            network.save_model(model_filename)

        elif choice == "0":
            break
        else:
            print("Invalid option, please try again")

def testing_menu(network):
    model_loaded = False  # Variable para verificar si el modelo ha sido cargado

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
            network.load_model(filename)
            model_loaded = True

        elif choice == "2":
            if not model_loaded:
                print("Error: No model loaded. Please load a model first (Option 1).")
            else:
                test_image_path = input("Enter the path to the image for testing: ")
                test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

                if test_image is None:
                    print("Error: Could not load the image")
                    continue

                # Mostrar la imagen original
                cv2.imshow("Original Image", test_image)

                # Realizar el preprocesamiento
                test_features = network.extract_features(test_image).reshape(1, -1)

                # Mostrar la imagen preprocesada (redimensionada)
                preprocessed_image = cv2.resize(test_image, (64, 128))  # Redimensiona a 64x128 para el HOG
                cv2.imshow("Preprocessed Image", preprocessed_image)

                # Esperar una tecla para cerrar las ventanas
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Realizar la predicción
                predicted_class = network.predict(test_features)
                predicted_letter = network.label_mapping.get(predicted_class[0], "Unknown")
                print(f"Predicted letter: {predicted_letter}")

        elif choice == "0":
            break
        else:
            print("Invalid option, please try again")

if __name__ == "__main__":
    main_menu()
