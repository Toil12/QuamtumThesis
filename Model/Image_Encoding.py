import pennylane as qml
from math import pi
import torch
from torch import nn
import time

class FRQI_Encoding(nn.Module):
    def __init__(self,qubits:int=3,input_pixels:int=4):
        super(FRQI_Encoding, self).__init__()
        self.n_qubits=qubits
        self.input_pixels=input_pixels
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self.frqi_layer=self.frqi_layers()

    def color_encoding(self,color):
        encoded_color=color/255*(pi/2)
        return encoded_color

    def forward(self,inputs):
        outputs=[]
        inputs=self.color_encoding(inputs)
        b,x,y=inputs.shape
        inputs=torch.reshape(inputs,(b,x*y))
        # print(inputs)
        for i in range(b):
            outputs.append(self.frqi_layer(inputs[i]))
        return outputs
        # return 0

    def mcy_gate(self,n_qubits):
        def ops(params):
            qml.RY(2 * params[0], wires=params[1])
        connection = qml.ctrl(ops, control=n_qubits - 2)
        s = reversed(list(range(0, n_qubits - 2)))
        for i in s:
            connection = qml.ctrl(connection, control=i)
        return connection

    def frqi_layers(self):
        weight_shapes = {}
        n_bits=self.n_qubits
        @qml.qnode(self.dev)
        def frqi_circuit(inputs):
            sections=len(inputs)
            for wire in range(n_bits - 1):
                qml.Hadamard(wires=wire)
            for i in range(sections):
                form='{'+f'0:0{n_bits-1}b'+'}'
                i_b=form.format(i)
                x_lines=[]
                for pos,bit in enumerate(i_b):
                    # print(pos,bit)
                    if bit=='0':
                        x_lines.append(pos)
                # set position
                for p in x_lines:
                    qml.PauliX(p)
                ops = self.mcy_gate(n_bits)
                ops(params=[inputs[i], n_bits-1])
                for p in x_lines:
                    qml.PauliX(p)
                # qml.Barrier()
            # return qml.state()
            # return qml.probs(wires=list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(0,n_bits )]
        # print(qml.draw(frqi_circuit)(torch.randint(0,2,(1,2**(self.n_qubits-1)))[0]))
        # print(qml.draw(frqi_circuit)(torch.randint(0, 2, (1, 7))[0]))
        q_compress_layer=qml.qnn.TorchLayer(frqi_circuit,weight_shapes)
        return q_compress_layer

if __name__ == '__main__':
    input=torch.randint(0,255,(32,3,3))
    model = torch.nn.Sequential(FRQI_Encoding(qubits=5))
    start=time.time()
    pred=model(input)
    end=time.time()
    print(end-start,"s")
    print(pred)

    # print(input)