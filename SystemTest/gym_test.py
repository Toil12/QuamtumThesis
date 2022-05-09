import pennylane as qml

dev = qml.device("default.qubit", wires=1, shots=10)

@qml.qnode(dev)
def circuit(x, y):
    qml.RX(x, wires=0)
    qml.RY(y, wires=0)
    return qml.expval(qml.PauliZ(0))

# execute the QNode using 10 shots
result = circuit(0.54, 0.1)


# execute the QNode again, now using 1 shot
result = circuit(0.54, 0.1, shots=1)
print(result)