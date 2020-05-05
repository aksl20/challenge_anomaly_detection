import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=0):
        self.last_metrics = 10**8
        self.patience = patience
        self.patience_count = 0

    def check_training(self, metric):
        if metric < self.last_metrics:
            stop_training = False
        elif (metric > self.last_metrics) & (self.patience_count < self.patience):
            self.patience_count += 1
            stop_training = False
        else:
            stop_training = True
        self.last_metrics = metric
        return stop_training

class PlotLoss:
    def __init__(self):
        self.losses = []

    def plot(self):
        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

class ModelManagement:
    def __init__(self, path, name_model):
        self.path = path
        self.last_metrics = 10**8
        self.name_model = name_model

    def save(self, model):
        torch.save(model.state_dict(), self.path + '%s' % self.name_model)

    def checkpoint(self, epoch, model, optimizer, loss):
        if self.last_metrics > loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, self.path+'%s_%d' %(self.name_model,epoch))

def loss_function(x, x_rec):
    return F.mse_loss(x, x_rec)

def train(model, data_loader, epoch, optimizer, callbacks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()

        x_rec = model.forward(data.float())
        loss = loss_function(data, x_rec)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx%25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx+1, len(data_loader),
                (batch_idx+1) / len(data_loader)*100, loss.item() / len(data)))
    avg_loss = train_loss / len(data_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    if (epoch % 10) == 0:
        callbacks['model_management'].checkpoint(epoch, model, optimizer, avg_loss)
    callbacks['plot_loss'].losses.append(avg_loss)
    return callbacks['early_stopping'].check_training(avg_loss)