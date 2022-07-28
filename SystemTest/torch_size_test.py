import torch
k=torch.tensor([[[1,2,3,4],[5,6,7,8]],[[1,1,1,1],[0,0,0,0]]])
# train_loader = torch.utils.data.DataLoader(
#     [[[1,2,3,4],[5,6,7,8]],[[1,1,1,1],[0,0,0,0]]], batch_size=2)
# for i in train_loader:
#    print(i)
k=k.resize(2,2)
print(k)