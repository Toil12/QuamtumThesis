import numpy as np
import pennylane as qml
import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN_Compress(nn.Module):
    def __init__(self,outfearue:int=4):
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
            nn.Linear(in_features=512, out_features=outfearue)
        )

    def forward(self, x):
        return self.fc(x)

class DQN_Q(nn.Module):
    def __init__(self,
                 a_size=3,
                 n_layers=1,
                 w_input=False,
                 w_output=False,
                 data_reupload: bool = False,
                 device_name:str="default.qubit",
                 encode_mode:int=0):
        super(DQN_Q, self).__init__()
        self.action_size=a_size
        self.n_qubits = 4
        self.n_layer=n_layers
        self.q_device = device_name
        self.encode_mode = encode_mode
        self.reupload = data_reupload

        self.q_layers=self.q_layer(self.n_layer,data_reupload)
        print(f"Encode mode is {self.encode_mode},device is {self.q_device}")

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

    def forward(self,inputs):
        #input shape 32*4
        if self.w_input is not None:
            inputs = inputs * self.w_input
        inputs = torch.atan(inputs)
        # bug place
        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2
        if self.w_output is not None:
            outputs = outputs * self.w_output
        else:
            outputs = 90 * outputs
        return outputs
        # outputs = self.q_layers(inputs)
        # return outputs

    def encode(self, inputs):
        # print(inputs)
        for wire in range(self.n_qubits):
            qml.RX(inputs[wire], wires=wire)

    def encode_dense(self,inputs):
        # print(self.n_qubits)
        for wire in range(self.n_qubits):
            x_2i_1=inputs[wire*2+1]*np.pi*2
            x_2i=inputs[wire*2]*np.pi*2
            qml.RY(x_2i_1,wires=wire)
            qml.RZ(x_2i,wires=wire)

    def layer(self, y_weight, z_weight):
        # print(y_weight)
        for wire, y_weight in enumerate(y_weight):
            qml.RY(y_weight, wires=wire)
            # print(qml.matrix(qml.RY(y_weight, wires=wire)))
        for wire, z_weight in enumerate(z_weight):
            qml.RZ(z_weight, wires=wire)
        for wire in range(self.n_qubits):
            qml.CZ(wires=[wire, (wire + 1) % self.n_qubits])

    def measure(self):
        return [qml.expval(qml.PauliZ(wire)) for wire in range(1,self.n_qubits)]

    def q_layer(self,n_layers, data_reupload):
        # dev = qml.device("lightning.qubit", wires=self.n_qubits)
        # dev = qml.device("default.qubit", wires=self.n_qubits)
        dev=qml.device(self.q_device,wires=self.n_qubits)
        weight_shapes = {"y_weights": (self.n_layer,self.n_qubits),
                         "z_weights": (self.n_layer,self.n_qubits)
        }

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, y_weights, z_weights):
            for layer_idx in range(n_layers):
                if (layer_idx == 0) or data_reupload:
                    # different encode mode
                    if self.encode_mode==0:
                        self.encode(inputs)
                    elif self.encode_mode==1:
                        self.encode_dense(inputs)
                self.layer(y_weights[layer_idx], z_weights[layer_idx])
            # print(self.measure())
            return self.measure()

        # @qml.qnode(dev,interface='torch')
        # def qnode(inputs, weights_0, weight_1):
        #     self.encode(inputs)
        #     qml.RX(0, wires=0)
        #     qml.RX(0, wires=1)
        #     qml.RX(0, wires=2)
        #     qml.RX(0, wires=3)
        #     return self.measure()\
        if self.encode_mode==0:
            print(qml.draw(circuit)([0.1,0.2,0.3,0.4],[[1,2,3,4]],[[1,2,3,4]]))
        elif self.encode_mode==1:
            print(qml.draw(circuit)([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],[[1,2,3,4,]],[[1,2,3,4]]))
        q_layers = qml.qnn.TorchLayer(circuit, weight_shapes)
        return q_layers



if __name__ == '__main__':
    # c=CNN_Compress()
    # q=DQN_Q()
    # model=torch.nn.Sequential(q)
    model=torch.nn.Sequential(CNN_Compress(),DQN_Q())
    pred=model(torch.tensor([[0,0,0,0]]*32))
    print(pred)




