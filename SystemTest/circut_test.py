import pennylane as qml
import math

shots_list = [5, 10, 1000]
dev = qml.device("default.qubit", wires=3)
@qml.qnode(dev)
def circuit(x,y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.CNOT(wires=[0, 2])
    return qml.probs(wires=[1,2]),qml.probs(wires=[0,1])

# print(qml.matrix(qml.RX(0.5*math.pi,0)))
# print(circuit(0.5*math.pi))
print(circuit(0.56,0.1))
print(qml.draw(circuit)(0.5*math.pi,0))


