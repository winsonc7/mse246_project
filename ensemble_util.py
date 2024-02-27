import numpy as np
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def ensemble_neural_network(X_train, y_train, B, num_neurons=64, activation='relu'):
    ensemble_models = []
    
    # Generate B bootstrap samples and train separate neural networks
    for b in range(B):
        print(b)
        # Create a bootstrap sample
        X_boot, y_boot = resample(X_train, y_train, replace=True)
        
        # Create a new neural network model
        model = Sequential()
        model.add(Dense(num_neurons, activation=activation, input_shape=(X_train.shape[1],)))
        model.add(Dense(num_neurons, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model on the bootstrap sample
        model.fit(X_boot, y_boot, epochs=10, batch_size=32, verbose=0)
        
        # Add the trained model to the ensemble
        ensemble_models.append(model)
    
    # Combine predictions of individual models by averaging
    def ensemble_predict(X):
        predictions = np.zeros((X.shape[0], len(ensemble_models)))
        for i, model in enumerate(ensemble_models):
            predictions[:, i] = model.predict(X).flatten()
        ensemble_prediction = np.mean(predictions, axis=1)
        return ensemble_prediction
    
    return ensemble_predict

"""
ensemble_predictor = ensemble_neural_network(X_train, y_train, B=10)
ensemble_predictions = ensemble_predictor(X_test)
"""