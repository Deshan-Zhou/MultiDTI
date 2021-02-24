class Config():
    def __init__(self):
        # model parmeters
        self.use_gpu = True

        self.repeat_nums = 1
        self.fold_nums = 10                 #The numbers of crossflod-validation
        self.neg_samp_ratio = 10            #NO. negative : NO. positive = nag_samp_ratio:1

        self.num_epochs = 30
        self.common_epochs = 1
        self.predict_epochs = 1
        self.batch_size = 128

        #Related to neural networks
        self.embedding_size = 64            # The embedding size for every word
        self.num_filters= 128               #The number of filter for convolutional layers  #[32,64]
        self.common_size = 32
        self.common_learn_rate = 0.00001    #Learning Rate
        self.pre_learn_rate = 0.00001

        #Related to the data set
        self.smi_n_gram = 1
        self.fas_n_gram = 3
        self.smi_dict_len=61                #The length of dictionary
        self.fas_dict_len=8083              #The length of dictionary
        self.fasta_max_len = 15000          #The max sequense length of protein
        self.smiles_max_len = 1500          #The max sequense length of smiles
        self.ds_nums = 5603
        self.se_nums = 4192

        # The path of data
        self.dg_ds_path = 'data/drug_disease.csv'
        self.dg_dg_path = 'data/drug_drug.csv'
        self.dg_pt_path = 'data/drug_protein.csv'
        self.dg_se_path = 'data/drug_se.csv'
        self.pt_ds_path = 'data/protein_disease.csv'
        self.pt_pt_path = 'data/protein_protein.csv'
        self.smi_dict_path = 'data/smi_dict.pickle'
        self.fas_dict_path = 'data/fas_dict.pickle'

        self.dg_smiles_path = 'data/durg_smiles.csv'
        self.pt_fasta_path = 'data/protein_fasta.csv'


