import pennylane as qml
import numpy as np

# 1. Define the number of features (which dictates the number of qubits)
N_FEATURES = 7

# 2. Initialize the quantum simulator
DEV = qml.device("default.qubit", wires=N_FEATURES)
print(f"[SETUP] Quantum simulator ready with {N_FEATURES} qubits")

# 3. Define the quantum circuit
@qml.qnode(DEV)
def quantum_feature_map(x):
    # Rotate each qubit based on the input features
    for i in range(N_FEATURES):
        qml.RY(x[i], wires=i) 
        
    # Create quantum entanglement between adjacent qubits
    for i in range(N_FEATURES - 1):
        qml.CNOT(wires=[i, i + 1]) 
        
    # Measure and return the expectation value (converts back to 7 real numbers)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_FEATURES)]

# 4. Test the circuit
if __name__ == "__main__":
    # Create a dummy array of 7 normalized values (between 0 and 1)
    # E.g., [soil_moisture, temp, humidity, light, N, P, K]
    sample_input = np.array([0.5, 0.8, 0.2, 0.9, 0.4, 0.3, 0.6])
    
    print("\n[INPUT] Classical data:")
    print(sample_input)
    
    print("\n[PROCESSING] Running data through quantum circuit...")
    quantum_output = quantum_feature_map(sample_input)
    
    print("\n[OUTPUT] Transformed quantum data:")
    # Convert to a standard numpy array for clean printing
    print(np.array(quantum_output))