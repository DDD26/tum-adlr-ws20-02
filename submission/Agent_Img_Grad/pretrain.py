import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3 import PPO, A2C, SAC

def pretrain_agent(student,
                   exp_train,
                   exp_test,
                   batch_size=64,
                   epochs=10,
                   scheduler_gamma=0.7,
                   learning_rate=1.0,
                   log_interval=100,
                   no_cuda=True,
                   seed=1,
                   test_batch_size=64):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.MSELoss()
    
    model = student.policy.to(device)
    def train(model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if isinstance(student, (A2C, PPO)):
                action, _, _ = model(data)
            else:
                # SAC/TD3:
                action = model(data)
            action_prediction = action
            loss = criterion(action_prediction, target)
            #loss = torch.nn.functional.cosine_embedding_loss(action_prediction,target,torch.ones(batch_size,dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,batch_idx * len(data),
                       len(train_loader.dataset),100.0 * batch_idx / len(train_loader),loss.item()))
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action
                loss = criterion(action_prediction, target)
                #test_loss = torch.nn.functional.cosine_embedding_loss(action_prediction,target,torch.ones(test_batch_size,dtype=torch.long).to(device))
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=exp_train, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=exp_test, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model