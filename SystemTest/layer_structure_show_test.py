import pennylane as qml


def encode(inputs):
    for wire in range(4):
        qml.RX(inputs[wire], wires=wire)

def measure():
    return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]

def layer(y_weight, z_weight):

    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(4):
        qml.CZ(wires=[wire, (wire + 1) % 4])

def get_layers(n_layers, data_reupload):
    dev = qml.device("default.qubit.tf", wires=4)
    shapes = {
        "y_weights": (n_layers, 4),
        "z_weights": (n_layers, 4)
    }

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            if (layer_idx == 0) or data_reupload:
                encode(inputs)
            layer(y_weights[layer_idx], z_weights[layer_idx])

        return measure()

    print(qml.draw(circuit)())

get_layers(5,True)