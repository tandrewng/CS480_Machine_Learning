Used CPU to train, will not work with cuda enabled. If you have cuda enabled change line 18
cuda = torch.cuda.is_available()
to 
cuda = not torch.cuda.is_available()