
from model import ResNet101,ResNet18
from optimizer import *
from engine import  *

from losses import criterion,KD,AT,SP
from data import train_loader,test_loader,val_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")















teacher = ResNet101().to(device)
teacher.load_state_dict(torch.load("checkpoints/teacher.pth", weights_only=True))

student = ResNet18().to(device)





optimizer = smart_optimizer(student, "AdamW", 0.0001, 0.9, 0.0005)
epochs = 100
warmup_steps = len(train_loader)*int(epochs*0.2)
stable_steps = len(train_loader)*int(epochs*0.1)
decay_steps = len(train_loader)*int(epochs*0.7)


scheduler = WarmupStableDecayLR( optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-5, base_lr=1e-3, final_lr=1e-5)




results = train(student,train_loader,test_loader,optimizer,criterion,epochs,scheduler,device)


"""results = train_kd(student,teacher,train_loader,test_loader,optimizer,criterion,KD,epochs,scheduler,device)
"""
"""results = train_AT(student,teacher,train_loader,test_loader,optimizer,criterion,SP,epochs,scheduler,device)
"""