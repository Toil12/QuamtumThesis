import pennylane as qml
import numpy as np
dev = qml.device('default.qubit', wires=1)
@qml.qnode(dev)
def my_quantum_function(x, y):
    # qml.Hadamard(wires=0)
    qml.RZ(x, wires=0)
    # return qml.state()
    return qml.expval(qml.PauliZ(0))


x=np.pi
y=0
circuit = qml.QNode(my_quantum_function, dev)
drawer = qml.draw(circuit, show_all_wires=True)
print(drawer(x,y))

r=circuit(x,y)
print(r)

