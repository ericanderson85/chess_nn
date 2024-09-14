import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from chess_neural_network import ChessNeuralNetwork
from chess_dataset import ChessDataset
from tqdm import tqdm

CONFIG = {
    'data_path': 'data.pt',
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 50,
    'validation_split': 0.1,
    'shuffle_dataset': True,
    'random_seed': 42,
    'save_model_path': 'chess_model.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}


def main():
    train_loader, val_loader = data_loaders(CONFIG)

    model = ChessNeuralNetwork().to(CONFIG['device'])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_mae = train(
            model, CONFIG['device'], train_loader, criterion, optimizer, epoch
        )

        val_loss, val_mae = validate(
            model, CONFIG['device'], val_loader, criterion, epoch
        )

        scheduler.step()

        print(f'Epoch {epoch+1}/{CONFIG["num_epochs"]}:')
        print(f'  Training Loss: {train_loss:.4f} | MAE: {train_mae:.4f}')
        print(f'  Validation Loss: {val_loss:.4f} | MAE: {val_mae:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['save_model_path'])
            print(f'  Saved model with validation loss: {best_val_loss:.4f}')

    print('Training complete.')
    print(f'Best Validation Loss: {best_val_loss:.4f}')


def data_loaders(config):
    dataset = ChessDataset(config['data_path'])

    total_size = len(dataset)
    val_size = int(total_size * config['validation_split'])
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle_dataset']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )

    return train_loader, val_loader


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f'Epoch {epoch+1} [Training]', leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(
            device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        mae = torch.abs(outputs - targets).sum().item()
        correct += mae
        total += inputs.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_mae = correct / total
    return epoch_loss, epoch_mae


def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f'Epoch {
                    epoch+1} [Validation]', leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(
                device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            mae = torch.abs(outputs - targets).sum().item()
            correct += mae
            total += inputs.size(0)

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_mae = correct / total
    return epoch_loss, epoch_mae


if __name__ == '__main__':
    main()
