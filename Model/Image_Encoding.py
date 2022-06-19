import numpy as np
import pennylane as qml
import torch
from torch import nn
from torch.nn.parameter import Parameter

class NEQR_Encoding:
    def __init__(self,qubits:int=10,input_size:int=1000):
        self.qubits=qubits+1
        self.input_size=input_size
    def encoding(self,color):
        for i in range(self.input_size):
            theta=color[i]/(np.pi/2)
            qml.CRY(theta,)


if __name__ == '__main__':
    pass