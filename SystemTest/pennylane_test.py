import pennylane as qml
dev = qml.device('lightning.qubit', wires=2, shots=1000)
print("finish")