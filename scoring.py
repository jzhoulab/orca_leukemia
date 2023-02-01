import selene_sdk
import pandas as pd
import numpy as np
import os
from selene_sdk.targets import Target
from selene_sdk.samplers import RandomPositionsSampler
import pyBigWig
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from scipy.stats import spearmanr
from selene_sdk.samplers.dataloader import SamplerDataLoader
from collections import OrderedDict
import sys
from selene_sdk.samplers.dataloader import SamplerDataLoader
plt_regl = False



variant = sys.argv[1] #'TL', 'DEL', 'DUP', 'INV'
cell_type_name = sys.argv[-3] #'T-ALL_GSE134761','THP1', 'GM12878', 'NALM6', 'Non-ETP_GSE146901','ETP_GSE146901', 'K562', 'KBM7'
df_name = sys.argv[-1] 
gene = sys.argv[-2] 


cell_type_normmat = {'T-ALL_GSE134761':0,
'THP1':1,
 'GM12878':2,
  'NALM6':3,
   'Non-ETP_GSE146901':4,
   'ETP_GSE146901':5,
   'K562':6,
    'KBM7':7}
normmat_files = ['./resources/GSE134761_TALL_all.hg38.no_filter.1000.mcool.expected.res4000.npy',
            './resources/THP1.hg38.no_filter.1000.mcool.expected.res4000.npy',
            './resources/4DNFIXP4QG5B.mcool.rebinned.mcool.expected.res4000.npy',
            './resources/NALM6.hg38.no_filter.1000.mcool.expected.res4000.npy',
            './resources/GSE146901_T_ALL_NonETP.hg38.no_filter.1000.mcool.expected.res4000.npy',
            './resources/GSE146901_T_ALL_ETP.hg38.no_filter.1000.mcool.expected.res4000.npy',
            './resources/GSE63525_K562.hg38.no_filter.1000.mcool.expected.res4000.npy',
            './resources/GSE63525_KBM7.hg38.no_filter.1000.mcool.expected.res4000.npy']

normmats_path = normmat_files[cell_type_normmat[cell_type_name]]
reg_files = [['./resources/Jurkat-H3K27ac.bw'],
['./resources/THP1_M.bw'],
['./resources/GM12878_H3K27ac_GSM733771.bw'],
['./resources/NALM6_H3K27ac_GSM3595838.bw'],
['./resources/Jurkat-H3K27ac.bw'],
['./resources/ETP-ALL_H3K27ac_GSM5330543.bw',
'./resources/KE37_H3K27ac_GSM5165987.bw'],
['./resources/K562.bw'],
['./resources/KBM7_H3K27ac_GSM4142031.bw']]
reg_path_file = reg_files[cell_type_normmat[cell_type_name]]
 
cell_type = cell_type_normmat[cell_type_name]

reg_path_file = np.loadtxt(reg_path_file, "str", ndmin=1)

expected_log_0 = np.load(normmats_path)
normmat = np.exp(expected_log_0[np.abs(np.arange(8000)[None, :] - np.arange(8000)[:, None])])
normmat_ch = np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)


class GenomicSignalFeatures(Target):
    def __init__(self, input_paths, features, shape):
        self.input_paths = input_paths
        self.initialized = False
        self.n_features = len(features)
        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)
        
    def get_feature_data(self, chrom, start, end, nan_as_zero=True):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            self.initialized=True

        wigmat = np.vstack([c.values(chrom, start, end)
                           for c in self.data])
        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat
    
def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])


    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

g = selene_sdk.sequences.Genome('./resources/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa')


t = GenomicSignalFeatures([reg_path for reg_path in reg_path_file],
[reg_path for reg_path in reg_path_file],
(100000,))


def PositionBinDiction(start, end, start_bin, chrm):
    bins={}
    for num, i in enumerate(range(start, end, bin_size)):
        
        bins[num+start_bin]=chrm+':'+str(i)
    return bins


def ReverseBinsOrder(diction, key_max):
    
    new_diction = {}
    for key in diction.keys():
        new_key = key_max-key
        new_diction[new_key] = diction[key]
    return new_diction
def ReverseBinsOrderStart(diction, key_min, key_max):
    new_diction = {}
    for key in diction.keys():
        new_key = key_min+(key_max-1)-key
        new_diction[new_key] = diction[key]
    return new_diction


def MoveBinsSetStart(diction, new_start_bin):
    new_diction = {}
    
    for num, key in enumerate(range(min(diction.keys()), (max(diction.keys())+1))):
        new_key = new_start_bin+num
        new_diction[new_key] = diction[key]
                
    return new_diction

def MoveBinsSetEnd(diction, new_end_bin):
    new_diction = {}
    
    dif = new_end_bin-max(diction.keys())
    
    for key in diction:
        
        new_key = key+dif
        new_diction[new_key] = diction[key]
    return new_diction

def DeleteExtraBins(diction, min_bin, max_bin):
    new_dicitonary = {}
    for key in diction:
        
        
        if min_bin<=key<max_bin:
            
            new_dicitonary[key] = diction[key]
        
    return new_dicitonary



def CopyEmptyVector(pos_vector):
    
    vector = {}
    for pos in pos_vector:
        vector[pos] = 999
    return vector
def ReverseKeyandValue(diciton):
    new_diction = {}
    for key in diciton:
        new_diction[diciton[key]] =  key
    return new_diction

def AddChromName(diction, chrm):
    new_diction = {}
    for key in diction:
        
        new_value = chrm+':'+str(diction[key])
        new_diction[key] = new_value
    return new_diction


def RoundPos(diction):
    new_diction = {}
    for key in diction:
        chrm, pos = key.split(':')
        
        pos = int(int(pos)/bin_size)*bin_size
        if (chrm+':'+str(pos)) in new_diction:
            new_diction[chrm+':'+str(pos)] = (new_diction[chrm+':'+str(pos)]+diction[key])/2
        else:    
            new_diction[chrm+':'+str(pos)] = diction[key]
    return new_diction

def RoundPosDup(diction):
    
    new_diction = {}
    dup_region = []
    
    for num, key in enumerate(diction):
        
        if 'dup' in key:
            chrm, pos, v = key.split(':')
        else:
            chrm, pos = key.split(':')
        pos = int(int(pos)/bin_size)*bin_size
        
        if ((chrm+':'+str(pos)) in new_diction) and ('dup' in key):
            
            new_diction[chrm+':'+str(pos)] = new_diction[chrm+':'+str(pos)]+diction[key]
            dup_region.append(chrm+':'+str(pos))
           
        else:    
            new_diction[chrm+':'+str(pos)] = diction[key]
            
    return new_diction, dup_region

def DeleteExtraPosition(diciton, min_pos, max_pos):
    min_pos = int(min_pos.split(':')[1])
    max_pos = int(max_pos.split(':')[1])
    new_dicition = {}
    for key in diciton:
        pos = int(key.split(':')[1])
        if min_pos<=pos<max_pos:
            new_dicition[key] = diciton[key]
    return new_dicition


chom_lengths = {}
chom_lengths_list = g.get_chr_lens()
for i in chom_lengths_list:
    chom_lengths['chr'+i[0]] = i[1]
    
gene_info = pd.read_csv('./resources/genes_lvl1_0.5threshold.csv')

def FindScorePlot(pos_key, pred, bin_num, gene_bin):
    
    chrm, pos = pos_key.split(':')
    

    if (chrm[-1] == 'a') or (chrm[-1] == 'b'):
        chrm = chrm[:-1]
    
    
    reg_signal= np.log(np.mean(np.mean(t.get_feature_data(chrm, int(pos), int(pos)+bin_size)[:], axis=1))+1)
    
    inter = pred[gene_bin][bin_num]*normmat_ch[gene_bin][bin_num]

    
    around_gene_bins = list(range(gene_bin-2, (gene_bin+2)+1))
    
    
    if bin_num in around_gene_bins:
        
        score = 0
        inter = 0
    else:
        score = inter*reg_signal 
        
    return score, reg_signal, inter

def FindScorePlotDup(pos_key, pred, bin_num, gene_bin, gene_bin2):
    
    chrm, pos = pos_key.split(':')
    
    if (chrm[-1] == 'a') or (chrm[-1] == 'b'):
        chrm = chrm[:-1]

    reg_signal= np.log(np.mean(np.mean(t.get_feature_data(chrm, int(pos), int(pos)+bin_size)[:], axis=1))+1)
    
    inter = pred[gene_bin][bin_num]*normmat_ch[gene_bin][bin_num]
 
    
    around_gene_bins1 = list(range(gene_bin-2, (gene_bin+2)+1))
    
    around_gene_bins2 = list(range(gene_bin2-2, (gene_bin2+2)+1))
    around_gene_bins = around_gene_bins1+around_gene_bins2
    if bin_num in around_gene_bins:
        
        score = 0
        inter = 0
    else:
        score = inter*reg_signal 
        
    return score, reg_signal, inter

def ScoreForEachBin(vector, pred, gene_bin, gene_bin2):
    
    inter_list = {}
    scores = {}
    reg = {}
    inter = {}
    for binn in  vector:
        pos = vector[binn]
        v=''
        if 'dup' in pos:
            p1, p2, v = pos.split(':')
            v=':'+v
            pos = p1+':'+p2
        if gene_bin2:
            scores[pos+v], reg[pos+v], inter[pos+v] = FindScorePlotDup(pos, pred, binn, gene_bin, gene_bin2)
        else:
            scores[pos+v], reg[pos+v], inter[pos+v] = FindScorePlot(pos, pred, binn, gene_bin)
           
    return scores, reg, inter

def SaveDF(lines, name):
    
    df = pd.DataFrame(lines, columns = ['name', 'score_gain', 'score_loss', 'score_both'])

    df_sorted = df.sort_values('score_gain')
    df_sorted = df_sorted.reset_index()

    df_sorted.to_csv(name)
    return df_sorted
def ScoresAllPredicitons(gene_bin_WT1, gene_bin_WT2, gene_bin_mutant1, gene_bin_mutant2,
                         vector_WT1, vector_WT2, vector_mutant,
                         pred_WT1, pred_WT2, pred_mutant):
    
    
  
    scores_WT1, reg_WT1, inter_WT1 = ScoreForEachBin(vector_WT1, pred_WT1, gene_bin=gene_bin_WT1, gene_bin2=None)
    scores_WT1 = RoundPos(scores_WT1)
    if plt_regl:
        reg_WT1 = RoundPos(reg_WT1)
        inter_WT1 = RoundPos(inter_WT1)
    
    gps = DeleteExtraBins(vector_WT1, gene_bin_WT1-2, gene_bin_WT1+2+1)
    gps = list(RoundPos(ReverseKeyandValue(gps)).keys())
    gps_mutant = DeleteExtraBins(vector_mutant, gene_bin_mutant1-2, gene_bin_mutant1+2+1)
    gps_mutant2 = []
    if variant == 'DUP':
        gps_mutant = list(RoundPosDup(ReverseKeyandValue(gps_mutant))[0].keys())
    else:
        gps_mutant = list(RoundPos(ReverseKeyandValue(gps_mutant)).keys())
    if gene_bin_mutant2:
        gps_mutant2 = DeleteExtraBins(vector_mutant, gene_bin_mutant2-2, gene_bin_mutant2+2+1)
        gps_mutant2 = list(RoundPosDup(ReverseKeyandValue(gps_mutant2))[0].keys())
   
 
    if (variant != 'TL') and (variant != 'INV'):
        if gene_bin_WT2:
            
            scores_WT2, reg_WT2, inter_WT2 = ScoreForEachBin(vector_WT2, pred_WT2, gene_bin=gene_bin_WT2, gene_bin2=None)
        else:
            scores_WT2 = {}
            if plt_regl:
                reg_WT2 = {} 
                inter_WT2 = {}
            for key in scores_WT2:
                scores_WT2[key] = 0
                if plt_regl:
                    reg_WT2[key] = 0
                    inter_WT2[key] = 0
            
        scores_WT2 = RoundPos(scores_WT2)
        if plt_regl:
            reg_WT2 = RoundPos(reg_WT2)
            inter_WT2 = RoundPos(inter_WT2)
  
        for key in scores_WT2:
            if (key in scores_WT1) and (key not in gps):
                scores_WT1[key] = (scores_WT1[key]+scores_WT2[key])/2
                if plt_regl:
                    reg_WT2[key] = (reg_WT2[key]+reg_WT2[key])/2
                    inter_WT2[key] = (inter_WT2[key]+inter_WT2[key])/2
            elif key not in scores_WT1:
                scores_WT1[key] = scores_WT2[key]
                if plt_regl:
                    reg_WT2[key] = reg_WT2[key]
                    inter_WT2[key] = inter_WT2[key]
    
    
    scores_WT = {}
    scores_WT.update(scores_WT1)
    if plt_regl:
        reg_WT = {}
        reg_WT.update(reg_WT1)
        inter_WT = {}
        inter_WT.update(inter_WT1)
    
    scores_mutant, reg_mutant, inter_mutant = ScoreForEachBin(vector_mutant, pred_mutant, gene_bin=gene_bin_mutant1, gene_bin2=gene_bin_mutant2)
   
    if plt_regl:
        reg_mutant = RoundPos(reg_mutant)
        inter_mutant = RoundPos(inter_mutant)
    if variant == 'DUP':
        scores_mutant, dup_region = RoundPosDup(scores_mutant)
    else:
        scores_mutant = RoundPos(scores_mutant)
        dup_region = []

    plotWT = []
    plotmutant = []
    for pos in scores_WT:
        if pos not in scores_mutant:

            scores_mutant[pos] = 0
            if plt_regl:
                try:
                    reg_mutant[pos] =reg_WT[pos]
                except:
                    None
                inter_mutant[pos] = 0
             
    for pos in scores_mutant:
        
        if pos not in scores_WT:
            
            scores_WT[pos] = 0
            if plt_regl:
                reg_WT[pos] =reg_mutant[pos]
                inter_WT[pos] = 0       
    if plt_regl:
        return scores_WT, scores_mutant, dup_region, reg_WT, reg_mutant, inter_WT, inter_mutant
    else:
         return scores_WT, scores_mutant, dup_region

def CreateGeneVectorDup(mbm, pbm, scores_WT, scores_mutant, vector_mutant, gene_pos_WT, gene_bin_mutant, reg_WT, reg_mutant, inter_WT, inter_mutant):

    gene_score_WT = DeleteExtraPosition(scores_WT, chrm1+':'+str(gene_pos_WT-496000), chrm1+':'+str(gene_pos_WT+496000))   
    if plt_regl:
        gene_reg_WT = DeleteExtraPosition(reg_WT, chrm1+':'+str(gene_pos_WT-496000), chrm1+':'+str(gene_pos_WT+496000))
        gene_inter_WT = DeleteExtraPosition(inter_WT, chrm1+':'+str(gene_pos_WT-496000), chrm1+':'+str(gene_pos_WT+496000))
        
    gene_window_bins = DeleteExtraBins(vector_mutant, gene_bin_mutant-62-mbm, gene_bin_mutant+62+pbm)
    
    def MutantGeneVector(gene_data, long_data):
        
        if variant == 'DUP':
            gene_data, dup_region = RoundPosDup(ReverseKeyandValue(gene_data))
        else:
            gene_data = RoundPos(ReverseKeyandValue(gene_data))
        gene_data2 ={}
        for key in gene_data:
            gene_data2[key] = long_data[key]
        gene_data = {}
        gene_data.update(gene_data2)
        return gene_data
    gene_score_mutant =  MutantGeneVector(gene_window_bins, scores_mutant)
    if plt_regl:
        gene_reg_mutant =  MutantGeneVector(gene_window_bins, reg_mutant)
        gene_inter_mutant =  MutantGeneVector(gene_window_bins, inter_mutant)
        

    def UnitVectors(gene_score_WT, gene_score_mutant):
        unit_WT_mutant = {}
        unit_WT_mutant.update(gene_score_WT)
        unit_WT_mutant.update(gene_score_mutant)
        return unit_WT_mutant
    unit_WT_mutant = UnitVectors(gene_score_WT, gene_score_mutant)
    if plt_regl:
        unit_WT_mutant_reg = UnitVectors(gene_reg_WT, gene_reg_mutant)
        unit_WT_mutant_inter = UnitVectors(gene_inter_WT, gene_inter_mutant)
    
    for key in unit_WT_mutant:

        if key not in gene_score_WT:
            
            gene_score_WT[key] = scores_WT[key]
            if plt_regl:
                gene_reg_WT[key] = reg_WT[key]
                gene_inter_WT[key] = inter_WT[key]

        
        if key not in gene_score_mutant:
           
            gene_score_mutant[key] = scores_mutant[key]
            if plt_regl:
                gene_reg_mutant[key] = reg_mutant[key]
                gene_inter_mutant[key] = inter_mutant[key]
    if plt_regl:
        return gene_score_WT, gene_score_mutant, gene_reg_WT, gene_reg_mutant, gene_inter_WT, gene_inter_mutant
    else:
        return gene_score_WT, gene_score_mutant

def FinalScore(gene_score_WT, gene_score_mutant):
    gene_score = np.zeros(len(gene_score_WT))
    for num, key in enumerate(gene_score_WT):
        gene_score[num] = gene_score_WT[key] - gene_score_mutant[key]
    score_loss = sum(gene_score[gene_score>0])
    score_gain = abs(sum(gene_score[gene_score<0]))

    score_both = score_loss+score_gain
    
    line = (gene_name, score_gain, score_loss, score_both)
    
    return line

bin_size = 8000
window = 1000000
level = 4 
breakpoint_bin = 126
max_bin = 250

print('variant: ', variant, flush=True)
lines = []
lines2 = []

try:
    SV = torch.load('prediction.pth')  
except:
    #no such a directory
    print('no prediciton file')
    sys.exit()
    


start_WT1 = SV[0]['start_coords'][level] 
start_WT2 = SV[1]['start_coords'][level] 
start_mutant1 = SV[2]['start_coords'][level] 
end_WT1 = SV[0]['end_coords'][level] 
end_WT2 = SV[1]['end_coords'][level] 

if (cell_type_name == 'T-ALL_GSE134761') or (cell_type_name == 'THP1'):
    m_type = 0
    if cell_type_name == 'T-ALL_GSE134761':
        cell_type = 0
    if cell_type_name =='THP1':
        cell_type = 1 
else:
    m_type = 1
    cell_type = cell_type-2


pred_WT1 = np.exp(SV[0]['predictions'][m_type][level][cell_type])
pred_WT2 = np.exp(SV[1]['predictions'][m_type][level][cell_type])
pred_mutant1 = np.exp(SV[2]['predictions'][m_type][level][cell_type])

if variant == 'TL':
   
    chrm1 = sys.argv[2] 
    breakpoint1 = int(sys.argv[3]) 
    chrm2 = sys.argv[4]
    breakpoint2 = int(sys.argv[5]) 
    strands = sys.argv[6] 
    chrm_len1 = chom_lengths[chrm1]
    chrm_len2 = chom_lengths[chrm2]

if variant == 'INV':
    chrm1 = sys.argv[2] 
    breakpoint1 = int(sys.argv[3]) 
    breakpoint2 = int(sys.argv[4]) 
    start_mutant2 = SV[3]['start_coords'][level]
    pred_mutant2 = np.exp(SV[3]['predictions'][m_type][level][cell_type])
    chrm_len1 = chom_lengths[chrm1]
    chrm_len2 = chrm_len1
if (variant == 'DEL') or (variant == 'DUP'):
    chrm1 = sys.argv[2] 
    breakpoint1 = int(sys.argv[3])
    breakpoint2 = int(sys.argv[4]) 
    chrm_len1 = chom_lengths[chrm1]
    chrm_len2 = chrm_len1


seq = g.get_encoding_from_coords(chrm1, breakpoint1-1000000, breakpoint1+1000000)
if np.sum(seq[:,0]==0.25) >= 200000:
    print('too many Ns')
    sys.exit()

if variant == 'TL':
    seq = g.get_encoding_from_coords(chrm2, breakpoint2-1000000, breakpoint2+1000000)
else:
    seq = g.get_encoding_from_coords(chrm1, breakpoint2-1000000, breakpoint2+1000000)

if np.sum(seq[:,0]==0.25) >= 200000:
    print('too many Ns')
    sys.exit()
    
border1 = (chrm_len1-128000-(breakpoint1-start_WT1))> breakpoint1 > (128000+(breakpoint1-start_WT1))
border2 = (chrm_len2-128000-(breakpoint2-start_WT2))> breakpoint2 > (128000+(breakpoint2-start_WT2))

if not border1 or not border2:
    print('too close to chrm border!')
    sys.exit()

if variant == 'TL':
    chrm1+='a'
    chrm2+='b'
 
Gene_names = {}

if gene not in Gene_names:


    try:
        TSS_pos_df = gene_info.loc[gene_info['geneNames'] == gene]
        
        
        for pos_i in TSS_pos_df.index:
            TSS_pos = TSS_pos_df['TSS_pos'][pos_i]
            Gene_names[(gene.split(',')[0])+';'+str(int(gene_info['expr_level'][pos_i]))] = TSS_pos

    
    except:
        print('unknown gene! gene is not int the CAGE TSS list')
        sys.exit()
        
if variant == 'TL':
    Gene_bins = {}

    for gene in Gene_names:
        Gene_bins[gene] = int((Gene_names[gene]-start_WT2)/bin_size)


    Gene_bins_left = {}
    for gene in Gene_bins:
        
        if (strands[1]=='+') and (62<=Gene_bins[gene]<breakpoint_bin):
            Gene_bins_left[gene] = Gene_bins[gene]
        elif (strands[1]=='-') and (max_bin-62>Gene_bins[gene]>=breakpoint_bin):
            Gene_bins_left[gene] = Gene_bins[gene]
    gene_bins_WT1 = {}
    gene_bins_WT1.update(Gene_bins_left)
    gene_bins_WT2 = {}


else:

    gene_bins_WT1 = {}
    gene_bins_WT2 = {}

    for gene in Gene_names:

        if ((variant == 'DEL') and ((Gene_names[gene]>=breakpoint2) or (Gene_names[gene]<=breakpoint1))) or (variant == 'INV') or (variant == 'DUP'):


            if 0<=(abs(breakpoint1-Gene_names[gene]))<int(window/2):
              
                gene_bins_WT1[gene] = int((Gene_names[gene]-start_WT1)/bin_size)

            if 0<=(abs(Gene_names[gene] - breakpoint2))<int(window/2):
               
                gene_bins_WT2[gene] = int((Gene_names[gene]-start_WT2)/bin_size)



vector_WT1 = PositionBinDiction(start_WT1, end_WT1, 0, chrm1)
if variant == 'TL':

    vector_WT2 = PositionBinDiction(start_WT2, end_WT2, 0, chrm2)
else:

    vector_WT2 = PositionBinDiction(start_WT2, end_WT2, 0, chrm1)


if variant == 'DUP':
    vector_m1 = DeleteExtraBins(vector_WT2, 0, breakpoint_bin)
    vector_m2 = DeleteExtraBins(vector_WT1, breakpoint_bin, max_bin)

    gene_bins_mutant1 = {}

    dub_region = int(np.rint((breakpoint2-breakpoint1)/bin_size))


    for key in gene_bins_WT2:

        if gene_bins_WT2[key]<breakpoint_bin:
            if (gene_bins_WT2[key]>=(breakpoint_bin-dub_region)) and (key in gene_bins_WT1):
                gene_bins_mutant1[key] = str(gene_bins_WT2[key])+'dub'+str(gene_bins_WT1[key])

            else:
                gene_bins_mutant1[key] = gene_bins_WT2[key]

    for key in gene_bins_WT1:
        if gene_bins_WT1[key]>=breakpoint_bin:
            if (gene_bins_WT1[key]<(breakpoint_bin+dub_region)) and (key in gene_bins_WT2):
                gene_bins_mutant1[key] = str(gene_bins_WT2[key])+'dub'+str(gene_bins_WT1[key])
            else:
                gene_bins_mutant1[key] = gene_bins_WT1[key]

    gene_bins_mutant2 = {}
    gene_bins_mutant2.update(gene_bins_mutant1)

   
    for key in vector_m1:    
        if key>=(breakpoint_bin-dub_region):
            vector_m1[key] = vector_m1[key]+':dup'

    for key in vector_m2:  
        if key<(breakpoint_bin+dub_region):
            vector_m2[key] = vector_m2[key]+':dup'


if variant == 'INV':
    inv_size = int((breakpoint2-breakpoint1)/bin_size)
    if inv_size >= 124:
        vector_m1 = DeleteExtraBins(vector_WT1, 0, breakpoint_bin)

        vector_m2 = DeleteExtraBins(vector_WT2, 0, breakpoint_bin)
        vector_m2 = ReverseBinsOrder(vector_m2, breakpoint_bin-1) 
        vector_m2 = MoveBinsSetStart(vector_m2, breakpoint_bin)
        vector_m2 = DeleteExtraBins(vector_m2, breakpoint_bin, max_bin)

        vector_2_m2 = DeleteExtraBins(vector_WT2, breakpoint_bin, max_bin)

        vector_2_m1 = DeleteExtraBins(vector_WT1, breakpoint_bin, max_bin)
        vector_2_m1 = ReverseBinsOrder(vector_2_m1, max_bin-1)
        vector_2_m1 = MoveBinsSetEnd(vector_2_m1, breakpoint_bin-1)
    else:
        vector_m1 = DeleteExtraBins(vector_WT1, 0, breakpoint_bin)

        vector_m1_1 = DeleteExtraBins(vector_WT1, breakpoint_bin, breakpoint_bin+inv_size)

        vector_m1_1 = ReverseBinsOrderStart(vector_m1_1, breakpoint_bin, breakpoint_bin+inv_size)


        vector_m1_2 = DeleteExtraBins(vector_WT1, breakpoint_bin+inv_size, max_bin)
        vector_m2 = {}
        vector_m2.update(vector_m1_1)
        vector_m2.update(vector_m1_2)

        vector_2_m1 = DeleteExtraBins(vector_WT2, breakpoint_bin, max_bin)
        vector_2_m2 = DeleteExtraBins(vector_WT2, 0, breakpoint_bin-inv_size)
        vector_1_m2 = DeleteExtraBins(vector_WT2, breakpoint_bin-inv_size, breakpoint_bin)
        vector_1_m2 = ReverseBinsOrderStart(vector_1_m2, breakpoint_bin-inv_size, breakpoint_bin)
        vector_2_m2.update(vector_1_m2)

    gene_bins_mutant1={}
    gene_bins_mutant2={}
    if inv_size >= 124:
        for key in gene_bins_WT1:
            if gene_bins_WT1[key]<breakpoint_bin:
                gene_bins_mutant1[key] = gene_bins_WT1[key]
            else:
                gene_bins_mutant2[key] = max_bin-gene_bins_WT1[key]+1

        for key in gene_bins_WT2:
            if gene_bins_WT2[key]>=breakpoint_bin:
                gene_bins_mutant2[key] = gene_bins_WT2[key]
            else:
                gene_bins_mutant1[key] = max_bin-gene_bins_WT2[key]+1
    else:
        for key in gene_bins_WT1:
            if (gene_bins_WT1[key]<breakpoint_bin) or (gene_bins_WT1[key]>=(breakpoint_bin+inv_size)):
                gene_bins_mutant1[key] = gene_bins_WT1[key]
            else:
                gene_bins_mutant1[key] = breakpoint_bin+(breakpoint_bin+inv_size)-gene_bins_WT1[key]

        for key in gene_bins_WT2:
            if (gene_bins_WT2[key]>=breakpoint_bin) or (gene_bins_WT2[key]<=(breakpoint_bin-inv_size)):
                gene_bins_mutant2[key] = gene_bins_WT2[key]
            else:
                gene_bins_mutant2[key] = (breakpoint_bin-inv_size)+(breakpoint_bin-1-gene_bins_WT2[key])

if variant == 'DEL':
    vector_m1 = DeleteExtraBins(vector_WT1, 0, breakpoint_bin)
    vector_m2 = DeleteExtraBins(vector_WT2, breakpoint_bin, max_bin)

    gene_bins_mutant1={}
    for key in gene_bins_WT1:
        if gene_bins_WT1[key]<breakpoint_bin:
            gene_bins_mutant1[key] = gene_bins_WT1[key]

    for key in gene_bins_WT2:
        if gene_bins_WT2[key]>=breakpoint_bin:
            gene_bins_mutant1[key] = gene_bins_WT2[key]
    gene_bins_mutant2 = {}
    gene_bins_mutant2.update(gene_bins_mutant1)



if variant == 'TL':
    


    if strands[0] == '+':

        ###take left part of WT2###
        vector_m1 = DeleteExtraBins(vector_WT1, 0, breakpoint_bin)




    elif strands[0] == '-':
        ###take the right part of WT2###
        vector_m1 = DeleteExtraBins(vector_WT1, breakpoint_bin, max_bin)

        ###in the mutant WT2 bins will be in the reverse order
        vector_m1 = ReverseBinsOrder(vector_m1, max_bin-1)

        ###the last bin for the left mutant vector will be breakpoint_bin-1###

        vector_m1 = MoveBinsSetEnd(vector_m1, breakpoint_bin-1)




    if strands[1] == '+':
        ###the left side of WT1###
        vector_m2 = DeleteExtraBins(vector_WT2, 0, breakpoint_bin)

        ###bins and their positions in the mutant will be reversed###
        vector_m2 = ReverseBinsOrder(vector_m2, breakpoint_bin-1) 
        ###the first bin of the right side in the mutant will be breakpoint_bin###
        vector_m2 = MoveBinsSetStart(vector_m2, breakpoint_bin)
        ###remove bins that go outside of predicted region###
        vector_m2 = DeleteExtraBins(vector_m2, breakpoint_bin, max_bin)

        #gene bins in the mutant
        gene_bins_mutant1 = {}
        for key in gene_bins_WT1:

            gene_bins_mutant1[key] = max_bin - gene_bins_WT1[key]+1

    elif strands[1] == '-':
        ###the right side of WT1###
        vector_m2 = DeleteExtraBins(vector_WT2, breakpoint_bin, max_bin)

        #gene bins in the mutant stay the same
        gene_bins_mutant1 = {}
        gene_bins_mutant1.update(gene_bins_WT1)



vector_m1.update(vector_m2)
vector_mutant1 = {}
vector_mutant1.update(vector_m1)

if variant == 'INV':
    vector_2_m1.update(vector_2_m2)
    vector_mutant2 = {}
    vector_mutant2.update(vector_2_m1)
if (variant == 'DUP') or (variant == 'DEL'):
    vector_mutant2 = {}
    vector_mutant2.update(vector_mutant1)



for gene_name in gene_bins_WT1:
    
    if (gene_name in gene_bins_mutant1) and (gene_name in gene_bins_WT1):
      

        gene_bin_mutant_list = gene_bins_mutant1[gene_name]
       
        try:
            gene_bin_mutant_list = gene_bin_mutant_list.split('dub')
        except:
            gene_bin_mutant_list = [gene_bin_mutant_list,]
       
        gene_bin_WT1 = gene_bins_WT1[gene_name]
        
        try:
            gene_bin_WT2 = gene_bins_WT2[gene_name]
        except:
            gene_bin_WT2 = ''

        gene_bin_mutant1 = int(gene_bin_mutant_list[0])
        if len(gene_bin_mutant_list)>1:
            gene_bin_mutant2 = int(gene_bin_mutant_list[1])
        else:
            gene_bin_mutant2=None

        dm = 0
        if len(gene_bin_mutant_list)>1:
            dm = int(gene_bin_mutant_list[1])-int((gene_bin_mutant_list[0]))


        if variant=='TL':

            if plt_regl:
                scores_WT, scores_mutant, dup_region, reg_WT, reg_mutant, inter_WT, inter_mutant = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT1, gene_bin_WT2 = gene_bin_WT2, gene_bin_mutant1=gene_bin_mutant1, gene_bin_mutant2=gene_bin_mutant2, 
                                                                vector_WT1=vector_WT2, vector_WT2=vector_WT1, vector_mutant =vector_mutant1, 
                                                                pred_WT1=pred_WT2, pred_WT2=pred_WT1, pred_mutant=pred_mutant1)


                gene_score_WT, gene_score_mutant, gene_reg_WT, gene_reg_mutant, gene_inter_WT, gene_inter_mutant = CreateGeneVectorDup(mbm=0, pbm=dm, 
                                                                    scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant1,
                                                                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant1,
                                                                    reg_WT=reg_WT, reg_mutant=reg_mutant, inter_WT=inter_WT, inter_mutant=inter_mutant)
            else:
                scores_WT, scores_mutant, dup_region = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT1, gene_bin_WT2 = gene_bin_WT2, gene_bin_mutant1=gene_bin_mutant1, gene_bin_mutant2=gene_bin_mutant2, 
                                                        vector_WT1=vector_WT2, vector_WT2=vector_WT1, vector_mutant =vector_mutant1, 
                                                        pred_WT1=pred_WT2, pred_WT2=pred_WT1, pred_mutant=pred_mutant1)

                gene_score_WT, gene_score_mutant = CreateGeneVectorDup(mbm=0, pbm=dm, 
                                                                    scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant1,
                                                                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant1,
                                                                reg_WT=None, reg_mutant=None, inter_WT=None, inter_mutant=None)
        else:
            if plt_regl:
                scores_WT, scores_mutant, dup_region, reg_WT, reg_mutant, inter_WT, inter_mutant = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT1, gene_bin_WT2 = gene_bin_WT2, gene_bin_mutant1=gene_bin_mutant1, gene_bin_mutant2=gene_bin_mutant2, 
                                                                vector_WT1=vector_WT1, vector_WT2=vector_WT2, vector_mutant =vector_mutant1, 
                                                                pred_WT1=pred_WT1, pred_WT2=pred_WT2, pred_mutant=pred_mutant1)

                gene_score_WT, gene_score_mutant, gene_reg_WT, gene_reg_mutant, gene_inter_WT, gene_inter_mutant = CreateGeneVectorDup(mbm=0, pbm=dm, 
                                                                    scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant1,
                                                                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant1,
                                                                    reg_WT=reg_WT, reg_mutant=reg_mutant, inter_WT=inter_WT, inter_mutant=inter_mutant)
            else:
                scores_WT, scores_mutant, dup_region = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT1, gene_bin_WT2 = gene_bin_WT2, gene_bin_mutant1=gene_bin_mutant1, gene_bin_mutant2=gene_bin_mutant2, 
                                                        vector_WT1=vector_WT1, vector_WT2=vector_WT2, vector_mutant =vector_mutant1, 
                                                        pred_WT1=pred_WT1, pred_WT2=pred_WT2, pred_mutant=pred_mutant1)

                gene_score_WT, gene_score_mutant = CreateGeneVectorDup(mbm=0, pbm=dm, 
                                                                    scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant1,
                                                                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant1,
                                                                reg_WT=None, reg_mutant=None, inter_WT=None, inter_mutant=None)



        if (variant == 'DUP') and (len(gene_bin_mutant_list)>1):

            
            scores_WT, scores_mutant, dup_region = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT1, gene_bin_WT2 = gene_bin_WT2, gene_bin_mutant1=gene_bin_mutant2, gene_bin_mutant2=gene_bin_mutant1,
                                                        vector_WT1=vector_WT1, vector_WT2=vector_WT2, vector_mutant =vector_mutant1, 
                                                        pred_WT1=pred_WT1, pred_WT2=pred_WT2, pred_mutant=pred_mutant1)


            gene_score_WT2, gene_score_mutant2 = CreateGeneVectorDup(mbm=dm, pbm=0, 
                                                                    scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant1,
                                                            gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant2,
                                                                    reg_WT=None, reg_mutant=None, inter_WT=None, inter_mutant=None)



            for num, key in enumerate(gene_score_mutant):
                gene_score_mutant[key] = gene_score_mutant[key]+gene_score_mutant2[key]


        line = FinalScore(gene_score_WT, gene_score_mutant)
        print(line)
        lines.append(line)


if variant != 'TL':
    for gene_name in gene_bins_WT2:

        if (gene_name in gene_bins_mutant2) and (gene_name in gene_bins_WT2):
           
            gene_bin_mutant_list = gene_bins_mutant2[gene_name]
            try:
                gene_bin_mutant_list = gene_bin_mutant_list.split('dub')
            except:
                gene_bin_mutant_list = [gene_bin_mutant_list,]

        
            gene_bin_WT2 = gene_bins_WT2[gene_name]
           
            try:
                gene_bin_WT1 = gene_bins_WT1[gene_name]
            except:
                gene_bin_WT1 = None

            gene_bin_mutant1 = int(gene_bin_mutant_list[0])
            if len(gene_bin_mutant_list)>1:
                gene_bin_mutant2 = int(gene_bin_mutant_list[1])
            else:
                gene_bin_mutant2=None

            
            dm = 0
            if len(gene_bin_mutant_list)>1:
                dm = int(gene_bin_mutant_list[1])-int((gene_bin_mutant_list[0]))
            if variant != 'INV':
                pred_mutant2 = np.copy(pred_mutant1)
            if plt_regl:
                scores_WT, scores_mutant, dup_region, reg_WT, reg_mutant, inter_WT, inter_mutant = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT2, gene_bin_WT2 = gene_bin_WT1, gene_bin_mutant1=gene_bin_mutant1, gene_bin_mutant2=gene_bin_mutant2,
                                                                vector_WT1=vector_WT2, vector_WT2=vector_WT1, vector_mutant =vector_mutant2, 
                                                                pred_WT1=pred_WT2, pred_WT2=pred_WT1, pred_mutant=pred_mutant2)

                gene_score_WT, gene_score_mutant, gene_reg_WT, gene_reg_mutant, gene_inter_WT, gene_inter_mutant = CreateGeneVectorDup(mbm=0, pbm=dm, 
                scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant2,
                                                                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant1,
                                                                    reg_WT=reg_WT, reg_mutant=reg_mutant, inter_WT=inter_WT, inter_mutant=inter_mutant)
            else:
                scores_WT, scores_mutant, dup_region= ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT2, gene_bin_WT2 = gene_bin_WT1, gene_bin_mutant1=gene_bin_mutant1, gene_bin_mutant2=gene_bin_mutant2,
                                                        vector_WT1=vector_WT2, vector_WT2=vector_WT1, vector_mutant =vector_mutant2, 
                                                        pred_WT1=pred_WT2, pred_WT2=pred_WT1, pred_mutant=pred_mutant2)

                gene_score_WT, gene_score_mutant = CreateGeneVectorDup(mbm=0, pbm=dm, 
                scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant2,
                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant1,
                reg_WT=None, reg_mutant=None, inter_WT=None, inter_mutant=None)

            if (variant == 'DUP') and (len(gene_bin_mutant_list)>1):


                scores_WT, scores_mutant, dup_region = ScoresAllPredicitons(gene_bin_WT1=gene_bin_WT2, gene_bin_WT2 = gene_bin_WT1, gene_bin_mutant1=gene_bin_mutant2, gene_bin_mutant2=gene_bin_mutant1,
                                                            vector_WT1=vector_WT2, vector_WT2=vector_WT1, vector_mutant =vector_mutant2, 
                                                            pred_WT1=pred_WT2, pred_WT2=pred_WT1, pred_mutant=pred_mutant2)


                gene_score_WT2, gene_score_mutant2 = CreateGeneVectorDup(mbm=dm, pbm=0,
                scores_WT=scores_WT, scores_mutant=scores_mutant, vector_mutant=vector_mutant2,
                                                                gene_pos_WT=Gene_names[gene_name], gene_bin_mutant=gene_bin_mutant2,
                                                                        reg_WT=None, reg_mutant=None, inter_WT=None, inter_mutant=None)



                for num, key in enumerate(gene_score_mutant):
                    gene_score_mutant[key] = gene_score_mutant[key]+gene_score_mutant2[key]



            line = FinalScore(gene_score_WT, gene_score_mutant)
            print(line)
            lines.append(line)

long_list = SaveDF(lines, df_name+'.csv')


##Create short list
def AddSVGenePairs(df):
    gene_names = []
    exprs = []
    for i in df.index:
        gene_name, expr = df['name'][i].split(';')
        gene_names.append(gene_name)
        exprs.append(expr)
    df['expr'] = exprs
    df['geneName'] = gene_names
    return df

def LeaveHighGainScore(df):
    droplines = []
    all_geneNames = list(OrderedDict.fromkeys(list(df['geneName'])))
    for gene_name in all_geneNames:
        
        f = df.loc[df['geneName'] == gene_name]
        f = f.sort_values(['score_gain'], ascending=False)
        
        max_score_ind = f.index[0]
        
        for i in f.index[1:]:
        
            droplines.append(i)
    df = df.drop(droplines)
    df = df.reset_index()
    df = df.drop(['level_0', 'index'], axis=1)
                
    return df

long_list = AddSVGenePairs(long_list)
short_list = LeaveHighGainScore(long_list)
short_list.to_csv(df_name+'.short.csv')
