import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class Model_pytorch(nn.Module) :
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )

    def forward(self, x) :
        return self.net(x)

    def train_pytorch(model, x_train, y_train, x_test, y_test, epochs=10, lr=0.003, batch_size=64) :
        optimizer = optim.Adam(model.parameters(), lr=lr)
        cross_entropy_loss = nn.CrossEntropyLoss()

        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size

        history = {"loss" : [] , "acc" : []}

        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        x_test_t = torch.tensor(x_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        for epoch in range(epochs) :
            model.train()
            indices = torch.randperm(num_samples)
            running_loss = 0.0

            for i in range(num_batches) :
                idx = indices[i * batch_size : (i + 1) * batch_size]
                x_batch , y_batch = x_train[idx] , y_train[idx]

                optimizer.zero_grad()
                scores = model(x_batch)
                loss = cross_entropy_loss(scores, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / num_batches
            history["loss"].append(epoch_loss)

            model.eval()
            with torch.no_grad() :
                test_scores = model(x_train_t)
                test_preds = torch.argmax(test_scores, dim=1)
                epoch_acc = (test_preds == y_test_t).float().mean().item() * 100
                history["acc"].append(epoch_loss)

        return history






        
