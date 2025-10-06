import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


if __name__ == '__main__':
    # Load the Wine dataset from sklearn
    wine = load_wine()
    X, y = wine.data, wine.target

    # For the purposes of this demo keep the UI small: pick first 4 features
    # Feature names (for reference): wine.feature_names[0:4]
    X = X[:, :4]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build a simple TensorFlow model appropriate for 3-class classification
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=2)

    model.save('my_model.keras')
    # Persist the scaler so the serving code can apply the same preprocessing
    joblib.dump(sc, 'scaler.pkl')
    print("Wine model was trained and saved")
