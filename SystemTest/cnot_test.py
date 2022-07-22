import pennylane as qml
from math import pi

dev = qml.device('default.qubit', wires=3)

def mcy_gate(n_qubits):
    def ops(params):
        qml.RY(2*params[0], wires=params[1])
    connection = qml.ctrl(ops, control=n_qubits - 2)
    s=reversed(list(range(0,n_qubits-2)))
    for i in s:
        connection = qml.ctrl(connection, control=i)

    return connection





@qml.qnode(dev)
def my_circuit(n_qubits:int=3,pixels:int=4,colors=[0,0,0,0]):
    for wire in range(n_qubits-1):
        qml.Hadamard(wires=wire)
    for i in range(pixels):
        # set position
        if i==0:
            pass
        elif i==1:
            qml.PauliX(1)
        elif i==2:
            qml.PauliX(0)
            qml.PauliX(1)
        elif i==3:
            qml.PauliX(1)

        # give color
        ops=mcy_gate(n_qubits)
        ops(params=[pi/2,2])
    # return qml.state()
    # return qml.probs(wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(wire)) for wire in range(0,n_qubits)]

print(qml.draw(my_circuit)())

r=my_circuit()
print(r)