import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, h):
        super(Net, self).__init__()

        # Shared Layers
        self.model = nn.Sequential(
            nn.Linear(1, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU()
        )
        # Sine branch
        self.model_sin = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )
        # Cosine Branch
        self.model_cos = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )

    def forward(self, inputs):
        # pass through shared layers
        x1 = self.model(inputs)

        # generate sin(x) prediction
        output_sin = self.model_sin(x1)

        # generate cos(x) prediction
        output_cos = self.model_cos(x1)

        # return both predictions
        return output_sin, output_cos

# Initialize the network
net = Net(150)

# Define loss function
loss_func = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Number of epochs
epochs = 10000

# Dummy input and output data (You should replace these with your actual data)
x = torch.randn(100, 1)  # 100 samples of 1D input
sin_true = torch.sin(x)  # Sine of the input
cos_true = torch.cos(x)  # Cosine of the input

for epoch in range(epochs):
    # Generate predictions
    sin_pred, cos_pred = net(x)

    # Compute loss
    loss1 = loss_func(sin_pred, sin_true)
    loss2 = loss_func(cos_pred, cos_true)

    # Add losses
    loss = loss1 + loss2

    # Run backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
