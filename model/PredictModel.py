from torch import nn
import torch
class Predict_model(nn.Module):

    def __init__(self):
        super(Predict_model,self).__init__()

        self.L_out = nn.Sequential(
            nn.Linear(1,1),
            nn.BatchNorm1d(1)
        )

    def forward(self,smi_common,fas_common,tag):
        bs = len(smi_common)
        predict = torch.sigmoid(self.L_out(torch.sum(torch.pow(torch.sub(smi_common,fas_common),2),dim=1).reshape(bs,1)))

        predict = predict.reshape(bs)

        return predict,tag
