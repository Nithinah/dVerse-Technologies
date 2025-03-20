# Install the required libraries and packages 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


# Load the Data

file_path = 'D:\\dVERSE\\parkinsons.data'  # Replace with actual path
parkinsons_data = pd.read_csv(file_path)

# Drop non-feature columns
features = parkinsons_data.drop(['name', 'status'], axis=1)
target = parkinsons_data['status']

# Scalarization 
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Reshaping the data to fit in the model
features_reshaped = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

# Split into train and test 
features_train, features_test, target_train, target_test = train_test_split(
    features_reshaped, target, test_size=0.2, random_state=42
)

# Data Augmentation 

def add_noise(data, noise_factor=0.02):
    noise = noise_factor * np.random.normal(size=data.shape)
    return data + noise

features_train_augmented = add_noise(features_train)

# Model + Keras Tuner

def build_model(hp):
    model = Sequential()

    # 1st CNN layer
    model.add(Conv1D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=16),
        kernel_size=hp.Int('kernel_size_1', min_value=3, max_value=7, step=2),
        activation='relu',
        input_shape=(features_reshaped.shape[1], 1)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    # 2nd CNN Layer
    model.add(Conv1D(
        filters=hp.Int('filters_2', min_value=32, max_value=128, step=16),
        kernel_size=hp.Int('kernel_size_2', min_value=3, max_value=7, step=2),
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

    # 3 Bidirection LSTM Layers
    # Layer 1
    model.add(Bidirectional(LSTM(
        units=hp.Int('lstm_units', min_value=32, max_value=128, step=16),
        return_sequences=True
    )))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.2, max_value=0.5, step=0.1)))

    # Layer 2
    model.add(Bidirectional(LSTM(
        units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=16)
    )))
    model.add(Dropout(hp.Float('dropout_lstm_2', min_value=0.2, max_value=0.5, step=0.1)))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dropout_dense', min_value=0.2, max_value=0.5, step=0.1)))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # model execution
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Keras Tuner

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='parkinsons_bidirectional'
)


# Hyperparameter Tuning

tuner.search(features_train_augmented, target_train, epochs=10, validation_split=0.2, batch_size=16)

# Best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Training the Model

optimized_model = tuner.hypermodel.build(best_hyperparameters)

history = optimized_model.fit(
    features_train_augmented, target_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Evaluating the Model

predictions = (optimized_model.predict(features_test) > 0.5).astype("int32")

# Accuracy
model_accuracy = accuracy_score(target_test, predictions)
print(f'\n Accuracy: {model_accuracy:.4f}')

# Classification Report
print("\n Classification Report:\n", classification_report(target_test, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(target_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
probabilities = optimized_model.predict(features_test).ravel()
false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, probabilities)
roc_auc_value = roc_auc_score(target_test, probabilities)

plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {roc_auc_value:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Decision Boundary
pca = PCA(n_components=2)
test_features_pca = pca.fit_transform(features_test.reshape(features_test.shape[0], -1))

x_min, x_max = test_features_pca[:, 0].min() - 1, test_features_pca[:, 0].max() + 1
y_min, y_max = test_features_pca[:, 1].min() - 1, test_features_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

decision_grid = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]).reshape(-1, features_test.shape[1], 1)
boundary_predictions = (optimized_model.predict(decision_grid) > 0.5).astype("int32").reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, boundary_predictions, alpha=0.6, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(test_features_pca[:, 0], test_features_pca[:, 1], c=target_test, cmap=ListedColormap(['#FF0000', '#0000FF']))
plt.title('Decision Boundary')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
