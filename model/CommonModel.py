from torch import nn
import torch
from torch.nn import functional as F

class Common_model(nn.Module):

    def __init__(self, config):
        super(Common_model, self).__init__()

        self.ds_common = nn.Parameter(torch.FloatTensor(config.ds_nums, config.common_size), requires_grad=True)
        self.se_common = nn.Parameter(torch.FloatTensor(config.se_nums, config.common_size), requires_grad=True)

        self.ds_common.data.normal_(0, 0.01)
        self.se_common.data.normal_(0, 0.01)

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.smi_emb = nn.Embedding(config.smi_dict_len + 1, config.embedding_size)
        self.smi_conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1)

        self.fas_emb = nn.Embedding(config.fas_dict_len + 1, config.embedding_size)
        self.fas_conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1)

        self.smi_mlp = MLP(config.num_filters,config.common_size)
        self.fas_mlp = MLP(config.num_filters,config.common_size)

    def forward(self, smiles,fasta):

        smiles_vector = self.smi_emb(smiles)                       #in:[batch_size,smi_max_len] out:[batch_size,smi_max_len,emb_size]

        smiles_vector = torch.unsqueeze(smiles_vector,1)           #out:[batch_size,1,smi_max_len,emb_size]

        smiles_vector = self.smi_conv_region(smiles_vector)        #out:[batch_size,num_filters,smi_max_len-3+1, 1]
        #Repeat 2 times
        smiles_vector = self.padding1(smiles_vector)               #out:[batch_size,num_filters,smi_max_len, 1]
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.conv(smiles_vector)                   #out:[batch_size,num_filters,smi_max_len-3+1, 1]
        smiles_vector = self.padding1(smiles_vector)               #out:[batch_size,num_filters,smi_max_len, 1]
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.conv(smiles_vector)                   #out:[batch_size,num_filters,smi_max_len-3+1, 1]

        while smiles_vector.size()[2] >= 2:
            smiles_vector = self._block(smiles_vector)
        smiles_vector = smiles_vector.squeeze()                    #[batch_size, num_filters]
        smile_common = self.smi_mlp(smiles_vector)                 #out:[smi_nums(batch),common_size]

        fasta_vector = self.fas_emb(fasta)                       #in:[smi_nums,smi_max_len] out:[smi_nums,smi_max_len,emb_size]

        fasta_vector = torch.unsqueeze(fasta_vector,1)           #out:[smi_nums,1,smi_max_len,emb_size]

        fasta_vector = self.fas_conv_region(fasta_vector)        #out:[batch_size,num_filters,smi_max_len-3+1, 1]
        #Repeat 2 times
        fasta_vector = self.padding1(fasta_vector)               #out:[batch_size,num_filters,smi_max_len, 1]
        fasta_vector = torch.relu(fasta_vector)
        fasta_vector = self.conv(fasta_vector)                   #out:[batch_size,num_filters,smi_max_len-3+1, 1]
        fasta_vector = self.padding1(fasta_vector)               #out:[batch_size,num_filters,smi_max_len, 1]
        fasta_vector = torch.relu(fasta_vector)
        fasta_vector = self.conv(fasta_vector)                   #out:[batch_size,num_filters,smi_max_len-3+1, 1]

        while fasta_vector.size()[2] >= 2:
            fasta_vector = self._block(fasta_vector)
        fasta_vector = fasta_vector.squeeze()                    #[batch_size, num_filters]
        fasta_common = self.fas_mlp(fasta_vector)

        return smile_common,fasta_common,self.ds_common,self.se_common

    def _block(self, x):
                                     #in: [batch_size,num_filters,smi_max_len-3+1, 1]
        x = self.padding2(x)         #out:[batch_size,num_filters,smi_max_len-1, 1]
        px = self.max_pool(x)        #out:[batch_size,num_filters,(smi_max_len-1)/2, 1]

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

# multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out