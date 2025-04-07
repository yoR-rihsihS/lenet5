import torch
import pickle
import matplotlib.pyplot as pyplot

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_definition import LeNet5
from loss_definition import Criterion

device = 'cuda'
torch.cuda.empty_cache()

def save_file(object, path):
    with open(path, 'wb') as file:
        pickle.dump(object, file)


def test(model, test_loader, loss_fn):
    correct = 0
    total = 0
    test_loss = 0
    model.eval()
    with torch.no_grad(): 
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            distances = model(images)
            loss = loss_fn(distances, labels)

            predictions = torch.argmin(distances.data, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
            test_loss += loss.item()

    return test_loss / len(test_loader), 100 * correct / total


def train(model, opt, sch, loss_fn, train_loader, test_loader, learning_rate, num_epochs):
    history = {'train_loss' : [], 'test_loss' : [], 'train_acc' : [], 'test_acc' : []}
    optimizer = opt(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = sch(optimizer, T_max=num_epochs, eta_min=0.000001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            distances = model(images)
            loss = loss_fn(distances, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = torch.argmin(distances.data, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
            train_loss += loss.item()

        # Test Step
        test_loss, test_acc = test(model, test_loader, loss_fn)
    
        scheduler.step()

        print(f"Epoch {epoch+1} \tTraining Loss : {train_loss / len(train_loader)} \tTest Loss : {test_loss}")
        print(f"Train Accuracy : {100 * correct / total} \tTest Accuracy : {test_acc}")
        print()

        history['train_loss'].append(train_loss / len(train_loader))
        history['test_loss'].append(test_loss)
        history['train_acc'].append(100 * correct / total)
        history['test_acc'].append(test_acc)

    return history


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
batch_size = 200

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model_1 = LeNet5(num_classes=10, conv_type='normal').to(device)
loss_fn = Criterion(j=0.3).to(device)
model_2 = LeNet5(num_classes=10, conv_type='depthwise-separable').to(device)
sch = torch.optim.lr_scheduler.CosineAnnealingLR
opt = torch.optim.SGD

history_1 = train(model_1, opt, sch, loss_fn, train_loader, test_loader, 0.003, 100)
torch.save(model_1.state_dict(), './saved/model_1.pth')
save_file(history_1, './saved/history_1.pkl')

history_2 = train(model_2, opt, sch, loss_fn, train_loader, test_loader, 0.003, 100)
torch.save(model_2.state_dict(), './saved/model_2.pth')
save_file(history_2, './saved/history_2.pkl')


def make_plot(train_values, test_values, x_label, y_label, legends, title, save_name):
    pyplot.plot(train_values, '-r')
    pyplot.plot(test_values, '-b')
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(legends)
    pyplot.title(title)
    pyplot.grid(True)
    pyplot.savefig('./saved/'+save_name+'.png')
    pyplot.close()

legends = ['Training Loss Convolution', 'Test Loss Convolution']
make_plot(history_1['train_loss'], history_1['test_loss'], 'epoch', 'loss', legends, 'Loss vs Number of Epochs', 'Normal_Loss_vs_Epoch.png')

legends = ['Training Accuracy Convolution', 'Test Accuracy Convolution']
make_plot(history_1['train_acc'], history_1['test_acc'], 'epoch', 'accuracy', legends, 'Accuracy vs Number of Epochs', 'Normal_Accuracy_vs_Epoch.png')

legends = ['Training Loss Depth-wise Separable Convolution', 'Test Loss Depth-wise Separable Convolution']
make_plot(history_2['train_loss'], history_2['test_loss'], 'epoch', 'loss', legends, 'Loss vs Number of Epochs', 'Depth_Loss_vs_Epoch.png')

legends = ['Training Accuracy Depth-wise Separable Convolution', 'Test Accuracy Depth-wise Separable Convolution']
make_plot(history_2['train_acc'], history_2['test_acc'], 'epoch', 'accuracy', legends, 'Accuracy vs Number of Epochs', 'Depth_Accuracy_vs_Epoch.png')