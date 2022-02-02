import json

from zmq import device

from nltk_utils import bag_of_words
from nltk_utils import tokenization,stem

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNet

with open ('AI_Chat_Bot\intents.json','r') as f:
    intents=json.load(f)


# list of all the words
all_words=[]
# classes 
tags=[]
# list of pattern and tags
xy=[]

for intent in intents['intents']:

    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenization(pattern)
        all_words.extend(w)
        xy.append((w,tag))


ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words ]
all_words=sorted(set(all_words))
tags=sorted(set(tags))

print(all_words)
print(tags)

x_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)

    label=tags.index(tag)
    y_train.append(label) # for the CrossEntropyLoss we need only class labels 
                          # not the one hot vectors 
    

x_train=np.array(x_train)
y_train=np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self) :
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples
    

# Hyperparameters
batch_size=8
input_size=len(all_words)
hidden_size=8
output_size=len(tags)
learning_rate=0.01
num_epochs=1000
#print(input_size,output_size)

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device)


#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for e in range(num_epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(dtype=torch.long).to(device)

    output=model(words)
    # if y would be one-hot, we must apply
    # labels = torch.max(labels, 1)[1]
    loss=criterion(output,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e+1) % 100==0:
        print(f'Epoch: {e+1}/{num_epochs},loss: {loss.item():.4f}')

print(f'Final loss={loss.item():.4f}')

data={
    'model_state': model.state_dict(),
    'input_size': input_size,
    'hidden_size':hidden_size,
    'output_size':output_size,
    'all_words': all_words,
    'tags': tags
}

file='data.pth'
torch.save(data,file)
print(f'training complete. file saved to {file}')
