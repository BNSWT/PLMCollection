import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from make_dataset import FunctionSet

input_size = 512*256*256
num_classes = 2
num_epochs = 100
batch_size = 100
learning_rate = 0.001

train_data = FunctionSet("data/Metal Ion Binding/train.txt", "output/alignments")
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model = nn.Linear(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (reprs, labels) in enumerate(train_loader):
        reprs = reprs.reshape(-1, input_size)
        
        outputs = model(reprs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))    