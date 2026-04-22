import torch
from model import HybridModel
from data_pipeline import build_dataset

model = HybridModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

X, Y = build_dataset(200)

for epoch in range(5):
    total_loss = 0

    for i in range(len(X)):
        inp = X[i].unsqueeze(0)
        target = Y[i].unsqueeze(0)

        output = model(inp)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X)}")

torch.save(model.state_dict(), "hybrid_model.pth")
