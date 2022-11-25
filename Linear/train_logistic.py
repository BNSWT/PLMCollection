import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data.dataloader import DataLoader
from make_dataset import FunctionSet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

input_size = 512*256*256
num_classes = 2
num_epochs = 100
batch_size = 100
learning_rate = 0.001


reprs_paths = ["data/Metal Ion Binding/reprs-0.npz.npy", "data/Metal Ion Binding/reprs-1.npz.npy", "data/Metal Ion Binding/reprs-2.npz.npy", "data/Metal Ion Binding/reprs-3.npz.npy", "data/Metal Ion Binding/reprs-4.npz.npy", "data/Metal Ion Binding/reprs-5.npz.npy"]
labels_paths = ["data/Metal Ion Binding/labels-0.npz.npy", "data/Metal Ion Binding/labels-1.npz.npy", "data/Metal Ion Binding/labels-2.npz.npy", "data/Metal Ion Binding/labels-3.npz.npy", "data/Metal Ion Binding/labels-4.npz.npy", "data/Metal Ion Binding/labels-5.npz.npy"]


# model = nn.Linear(input_size, num_classes)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# for epoch in range(num_epochs):
#     for path_index in range(len(reprs_paths)):
#         train_data = FunctionSet(None, None, repr_path=reprs_paths[path_index], label_path=labels_paths[path_index])
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  
#         total_step = len(train_loader)*len(reprs_paths)
#         for i, (reprs, labels) in enumerate(train_loader):
#             reprs = reprs.reshape(-1, input_size)
            
#             outputs = model(reprs)
#             loss = criterion(outputs, labels)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             _, predicted = torch.max(outputs.data, 1)
#             total = labels.size(0)
#             correct = (predicted == labels).sum()
            
#             print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.4f},  Loss: {:.4f}' 
#                     .format(epoch+1, num_epochs, i+1, total_step, correct/total, loss.item()))    

# torch.save(model.state_dict(), 'linear_model.ckpt')

      
model = nn.Linear(input_size, num_classes)
checkpoint = torch.load('linear_model.ckpt')
model.load_state_dict(checkpoint)
  
test_reprs_paths = ["data/Metal Ion Binding/test-reprs-0.npz.npy", "data/Metal Ion Binding/test-reprs-1.npz.npy"] 
test_labels_paths = ["data/Metal Ion Binding/test-labels-0.npz.npy", "data/Metal Ion Binding/test-labels-1.npz.npy"] 

with torch.no_grad():
    correct = 0
    total = 0
    for path_index in range(len(test_reprs_paths)):
        test_data = FunctionSet(None, None, repr_path=test_reprs_paths[path_index], label_path=test_labels_paths[path_index])
        test_loader = DataLoader(test_data, batch_size=input_size, shuffle=False)
        for reprs, labels in test_loader:
            reprs = reprs.reshape(-1, input_size)
            outputs = model(reprs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('Accuracy of the model: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'linear_model.ckpt')