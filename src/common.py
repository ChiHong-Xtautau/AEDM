import torch


def accuracy(net, test_loader, device):

    net.eval()

    total = 0.0
    correct = torch.tensor(0.0).to(device)
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted = predicted.to(device)
            correct += (predicted == labels).sum()

    return 100. * float(correct) / total
