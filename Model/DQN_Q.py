from torch import nn
from torch.nn.parameter import Parameter
from torchsummary import summary

import torch
import pennylane as qml


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN_Compress(nn.Module):
    def __init__(self):
        super(CNN_Compress, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4)
        )

    def forward(self, x):
        return self.fc(x)


class DQN_Q(nn.Module):
    def __init__(self, a_size=3, n_layers=1, w_input=False, w_output=False, data_reupload: bool = True):
        super(DQN_Q, self).__init__()
        self.n_qubits = 4
        self.n_actions = a_size
        self.data_reupload = data_reupload
        self.q_layers = self.get_layers(n_layers=n_layers, data_reupload=data_reupload)

        if w_input:
            self.w_input = Parameter(torch.Tensor(self.n_qubits))
            nn.init.normal_(self.w_input)
        else:
            self.register_parameter('w_input', None)
        if w_output:
            self.w_output = Parameter(torch.Tensor(self.n_actions))
            nn.init.normal_(self.w_output, mean=90.)
        else:
            self.register_parameter('w_output', None)

    def forward(self, inputs):
        # input shape 32*4
        if self.w_input is not None:
            inputs = inputs * self.w_input
        inputs = torch.atan(inputs)
        # bug place
        outputs = self.q_layers(inputs)


        print("check")
        outputs = (1 + outputs) / 2
        if self.w_output is not None:
            outputs = outputs * self.w_output
        else:
            outputs = 90 * outputs
        return outputs

    def encode(self, inputs):
        for wire in range(self.n_qubits):
            qml.RX(inputs[wire], wires=wire)


    def layer(self, y_weight, z_weight):

        for wire, y_weight in enumerate(y_weight):
            qml.RY(y_weight, wires=wire)
            # print(qml.matrix( qml.RY(y_weight, wires=wire)))
        for wire, z_weight in enumerate(z_weight):
            qml.RZ(z_weight, wires=wire)
        for wire in range(self.n_qubits):
            qml.CZ(wires=[wire, (wire + 1) % self.n_qubits])

        # print(qml.state())




    def measure(self):
        # return [
        #     qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        #     qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
        # ]

        return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def get_layers(self, n_layers, data_reupload):
        dev = qml.device("default.qubit.tf", wires=self.n_qubits)
        shapes = {
            "y_weights": (n_layers, self.n_qubits),
            "z_weights": (n_layers, self.n_qubits)
        }

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, y_weights, z_weights):
            for layer_idx in range(n_layers):
                if (layer_idx == 0) or data_reupload:
                    self.encode(inputs)
                self.layer(y_weights[layer_idx], z_weights[layer_idx])

            return self.measure()

        q_layers = qml.qnn.TorchLayer(circuit, shapes)
        # print(qml.draw(circuit)((n_layers, self.n_qubits),(n_layers, self.n_qubits)))

        return q_layers
