import torch
class Helper():
    """
    Tools
    """
    def __init__(self):
        self.a = None

    def to_longtensor(self,x,use_gpu):
        x = torch.LongTensor(x)
        if use_gpu:
            x = x.cuda()
        return x

    def to_floattensor(self,x,use_gpu):
        x = torch.FloatTensor(x)
        if use_gpu:
            x = x.cuda()
        return x

    def comput_distance_loss(self,smi_common,fas_common,tag,dg_index,pt_index,ds_common,se_common,dg_dg,dg_se,dg_ds,pt_pt,pt_ds):

        dg_dg = dg_dg[:,dg_index]
        pt_pt = pt_pt[:,pt_index]

        total_loss = torch.tensor(0.0).cuda()
        #dg-pt interaction loss
        dg_pt_temp1 = torch.pow(torch.sub(smi_common,fas_common),2)
        dg_pt_temp2 = torch.sum(dg_pt_temp1,dim=1)
        total_loss += torch.sum(torch.mul(dg_pt_temp2,tag))

        for i in range(dg_index.shape[0]):
            #dg-dg interaction
            dg_dg_temp1 = torch.pow(torch.sub(smi_common[i,:],smi_common),2)
            dg_dg_temp2 = torch.mul(dg_dg_temp1.t(),dg_dg[dg_index[i],:])
            total_loss += torch.sum(dg_dg_temp2)

            #dg-ds association
            dg_ds_temp1 = torch.pow(torch.sub(smi_common[i,:],ds_common),2)
            dg_ds_temp2 = torch.mul(dg_ds_temp1.t(),dg_ds[dg_index[i],:])
            total_loss += torch.sum(dg_ds_temp2)

            #dg-se association
            dg_se_temp1 = torch.pow(torch.sub(smi_common[i,:],se_common),2)
            dg_se_temp2 = torch.mul(dg_se_temp1.t(),dg_se[dg_index[i],:])
            total_loss += torch.sum(dg_se_temp2)

        for i in range(pt_index.shape[0]):
            #pt-pt interaction
            pt_pt_temp1 = torch.pow(torch.sub(fas_common[i,:],fas_common),2)
            pt_pt_temp2 = torch.mul(pt_pt_temp1.t(),pt_pt[pt_index[i],:])
            total_loss += torch.sum(pt_pt_temp2)
            #pt-ds association
            pt_ds_temp1 = torch.pow(torch.sub(fas_common[i,:],ds_common),2)
            pt_ds_temp2 = torch.mul(pt_ds_temp1.t(),pt_ds[pt_index[i],:])
            total_loss += torch.sum(pt_ds_temp2)

        return total_loss






