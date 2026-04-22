import torch
import torch.nn as nn
import pennylane as qml

# Quantum setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_layer(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(8*32*32, 4)

        # Quantum weights
        self.q_weights = nn.Parameter(torch.randn(3, n_qubits, 3))

        self.fc2 = nn.Linear(4, 8*32*32)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Quantum layer (batch loop)
        q_out = []
        for i in range(x.shape[0]):
            q_out.append(torch.tensor(quantum_layer(x[i], self.q_weights)))

        x = torch.stack(q_out)

        x = self.fc2(x)
        x = x.view(-1, 8, 32, 32)

        x = self.decoder(x)
        return x
