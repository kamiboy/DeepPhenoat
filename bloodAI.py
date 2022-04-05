#A program training AI prediction models based on binary phenotypes and specified genotypes
#Camous Moslemi, 05.04.2022

import os
from os.path import exists
from datetime import datetime
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from bedbug import BEDBUG
from bedbug import Variant

version = 3.5

perform_training = False
perform_prediction = True

#Pick up from where the work was left off
resume = False

#Extend will increase the considered genomic area by the extension number of basepairs up and downstream of provided locus
extend = True
extension = 20000

#Activate MLP or AEC training or preiction
MLP = True
AEC = True

#dataset split for training and validation, test set will be total - (train + validate) if train and validation add to to 1 then test will just be validation
data_rate_train = 0.75
data_rate_validate = 0.25

#Limit hours spent training a model to set amount, when reached the training will be interrupted and whatever was the best achieved model by then will be used
limitHours = False
maxTrainingHours = 24

#An accuracy threshold below which model training is considered stuck in a saddle point if not surpassed after a set amount of epochs, traiggering restart
saddle_accuracy = 0.9
min_accuracy = 0.92
min_restarts_mlp = 5
max_restarts = 15
retries_mlp = 100
retries_aec = 15

#the factor with which to reduce AEC model Autoencoder bottleneck relative to input/output, and a minumum acceptable dimension
dim_factor = 0.05
min_dim = 4
#the factor with which to reduce MLP model deep layer size relative to input, and a minumum acceptable deep layer size
hid_factor = 0.1
min_hid = 4

#A list of phenotypes to be skipped for training or predicting
skipped = []

#folder where bed/bim/fam genotype files are stored
snpdir = '/snpdir/'

#If each chromosome is in a separate bed file
#chr_file = '20210503_chr'
#chr_file_postfix = '_CVD_FINAL'
#If all chromosomes are in one bed file
chr_file = 'CVD_GSA_FINAL'
chr_file_postfix = None

#directory where all data files except for genotypes are loaded from and written to
workdir = '/workdir/'

#File specifying genomic loci for each phenotype for model training and/or prediction  
phenotypes_info_file = 'phenotypes.info.tsv'
phenotypes_file = 'phenotypes.tsv'

class Progress:
    def __init__(self, iterations, steps = 20):
        self.steps = steps
        self.progress = 0
        self.iterations = iterations
        print('progress:',end=' ')

    def step(self,step=1):
        self.progress += step
        if round((self.progress-step) /(self.iterations/self.steps)) != round((self.progress)/(self.iterations/self.steps) ):
            print('%i'%(self.progress*100/self.iterations), end='% ', flush=True)
            #print('#',end='', flush=True)

class Case:
    def __init__(self, id):
        self.id = id
        self.phenotypes = {}
        self.sources = {}
        self.conflicts = set()
        self.typed = {}

    def type(self, phenotype, status, source):

        if phenotype == '*':
            return

        if phenotype in self.phenotypes:
            self.typed[phenotype] = self.typed[phenotype] + 1
            self.sources[phenotype].add(source)
            if self.phenotypes[phenotype] != status:
                self.conflicts.add(phenotype)
        else:
            self.phenotypes[phenotype] = status
            self.typed[phenotype] = 1
            self.sources[phenotype] = {source}

class Phenotyper:
    def __init__(self):
        self.cases = {}
        self.nPhenotypes = 0

    def load(self, file):
        if file == None:
            print('No phenotypes file has been provided.')
            return
           
        print('Loading phenotypes.')
        file = open(file, "r")
        linenr = 0
        for line in file:
            linenr += 1

            if linenr == 1:
                continue

            items = line.rstrip().split('\t')

            id = items[0]
            phenotype = items[1]
            status = (items[2] == '1')

            if not id in self.cases:
                self.cases[id] = Case(id)
            case = self.cases[id]

            case.type(phenotype, status, "serologi")
        self.nPhenotypes = self.nPhenotypes + linenr - 1
        print("Number of phenotyped cases: " + str(len(self.cases)))
        print("Number of phenotypes: " + str(linenr-1))

class AIDataset(Dataset):
    def __init__(self, phenotype, loci, pos, neg, mutationrate = 0, maxmissing=1, specific_variants=None):
        PYTHONHASHSEED=0
        self.mutationrate = mutationrate
        self.train_mode = True

        #Is dataset for training a phenotype prediction model, or for phenotype prediction based pre-trained model
        if pos == None or neg == None:
            self.train_mode = False

        if self.train_mode:
            self.pos = set(pos)
            self.neg = set(neg)
            self.rateComplete = 0
            self.rateCompletePos = 0
            self.rateCompleteNeg = 0
            print('Loading genotypes for training on %i (+) and %i (-) %s phenotypes.'%(len(self.pos),len(self.neg),phenotype) )
        else:
            print('Loading genotypes for predicting %s phenotypes'%(phenotype))

        #bed = BEDBUG('/data/preprocessed/genetics/dbds_freeze_20210503/DBDS_GSA_FINAL')
        #bed = BEDBUG('/data/preprocessed/genetics/chb_cvd_freeze_20210503/CVD_GSA_FINAL')

        first = True
        for locus in loci:
            locus = locus.split(':')
            chr = locus[0][3:]
            locus = locus[1].split('-')
            start = int(locus[0])
            end = int(locus[1])
            if extend:
                start = start - extension
                end = end + extension
                if start < 0:
                    start = 0

            if chr_file_postfix == None:
                bed = BEDBUG(snpdir+chr_file)
            else:
                bed = BEDBUG(snpdir+chr_file+chr+chr_file_postfix)

            if self.train_mode:
                (vars,cases,_,geno) = bed.region(chr, start, end, self.pos|self.neg)
            else:
                (vars,cases,_,geno) = bed.region(chr, start, end, [])

            if len(vars) == 0:
                print('Warning: No genotypes found at ' + chr + ':' + str(start) + '-' + str(end) )

            if first:
                self.variants = vars
                self.cases = cases
                genotypes = geno
                first = False
            else:
                self.variants += vars
                if self.cases != cases:
                    print('Error: cases do not match across loci!')
                    exit(1)
                genotypes += geno

        if specific_variants != None:
            variants = self.variants
            cases = self.cases

            self.variants = []
            specific_variants = open(specific_variants,'r')
            added = [False]*len(variants)
            data = []
            header = True
            missing = 0
            for line in specific_variants:
                if header:
                    header = False
                    continue

                items = line.rstrip().split('\t')
                chr = items[0]
                pos = int(items[1])
                allele1 = items[2]
                allele2 = items[3]

                found = False
                for index in range(len(variants)):
                    if added[index]:
                        continue

                    if variants[index].chr == chr and variants[index].pos == pos and variants[index].allele1 == allele1 and variants[index].allele2 == allele2:
                        data += genotypes[index*len(cases):(index+1)*len(cases)]
                        found = True
                        added[index] = True
                        self.variants.append(variants[index])
                        break

                if not found:
                    print('Warning: %s variant %s:%i%s/%s not found, this could reduce prediction accuracy'%(phenotype,chr,pos,allele1,allele2))
                    missing += 1
                    if not self.train_mode:
                        data += [-1]*len(cases)
                        self.variants.append(Variant(None,chr,pos,allele1,allele2))
            specific_variants.close()

            print('Found %i/%i (%.2f%%) specified variants in dataset'%(len(self.variants)-missing,len(self.variants), (len(self.variants)-missing)*100/len(self.variants)))
            del(genotypes)
            self.cases = cases
            self.data = np.array(data,dtype='int8')
            self.data = self.data.reshape((len(self.variants), len(self.cases)))
            self.data = np.transpose(self.data)
            del(data)
        else:
            #Keep only variants with missingness below threhold for training
            if self.train_mode:
                keep = []
                variants = []
                for index in range(len(self.variants)):
                    if self.variants[index].na < maxmissing:
                        variants.append(self.variants[index])
                        keep.append(index)
            else:
                 variants = self.variants

            print('Found %i genotypes for %i cases.'%(len(variants),len(self.cases)))

            if len(variants) == 0 or len(self.cases) == 0:
                print('Error: %s has insufficient variants (%i) or cases (%i)'%(phenotype, len(variants),len(self.cases)))
                return

            self.data = np.array(genotypes,dtype='int8')
            self.data = self.data.reshape((len(self.variants), len(self.cases)))
            if self.train_mode:
                self.data = self.data[keep]
            self.data = np.transpose(self.data)
            self.variants = variants

            del(genotypes)

        if not self.train_mode:
            self.phenotypes = []
            self.pos = set()
            self.neg = set()
            self.allhash = set()
            self.poshash = set()
            self.neghash = set()
            self.completeposhash = set()
            self.completeneghash = set()
            self.rateComplete = 0
            self.rateCompletePos = 0
            self.rateCompleteNeg = 0
            return

        #Assign phenotype to each genotyped case
        self.phenotypes = []
        pos = set()
        neg = set()
        for id in self.cases:
            if id in self.pos:
                pos.add(id)
                self.phenotypes.append(1)
            elif id in self.neg:
                neg.add(id)
                self.phenotypes.append(0)
            else:
                print('Error: Unknown ID ' + id)
                #exit(1)
        self.pos = pos
        self.neg = neg
        
        self.allhash = set()
        self.completeposhash = set()
        self.completeneghash = set()
        self.poshash = set()
        self.neghash = set()
        posdict = {}
        negdict = {}
        conflicthash = set()
        conflictposhash = set()
        conflictneghash = set()

        nConflict = 0
        nConflictPos = 0
        nConflictNeg = 0

        nComplete = 0
        nIncomplete = 0
        nCompletePos = 0
        nIncompletePos = 0
        nCompleteNeg = 0
        nIncompleteNeg = 0

        for index in range(len(self.data)):
            missing = False
            if self.data[index].min() < 0:
                missing = True
            if missing:
                nIncomplete += 1
            else:
                nComplete += 1

            h = hash(self.data[index].tostring())
            #h = hash(self.data[index].tobytes()
            self.allhash.add(h)
            if self.phenotypes[index] == 1:
                if missing:
                    nIncompletePos += 1
                else:
                    nCompletePos += 1
                    self.completeposhash.add(h)
                self.poshash.add(h)
                if h in self.neghash:
                    conflicthash.add(h)
                    conflictposhash.add(h)
                    nConflict = nConflict + 1
                    nConflictPos = nConflictPos + 1

                if h in posdict:
                    posdict[h] = posdict[h] + 1
                else:
                    posdict[h] = 1
            else:
                if missing:
                    nIncompleteNeg += 1
                else:
                    nCompleteNeg += 1
                    self.completeneghash.add(h)
                self.neghash.add(h)
                if h in self.poshash:
                    conflicthash.add(h)
                    conflictneghash.add(h)
                    nConflict = nConflict + 1
                    nConflictNeg = nConflictNeg + 1

                if h in negdict:
                    negdict[h] = negdict[h] + 1
                else:
                    negdict[h] = 1

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for h in posdict:
            if h in conflicthash:
                if posdict[h] >= negdict[h]:
                    TP = TP + posdict[h]
                    FN = FN + negdict[h]
            else:
                TP = TP + posdict[h]

        for h in negdict:
            if h in conflicthash:
                if negdict[h] > posdict[h]:
                    TN = TN + negdict[h]
                    FP = FP + posdict[h]
            else:
                TN = TN + negdict[h]

        recall = 0
        specificity = 0 
        PPV = 0
        NPV = 0
        accuracy = 0

        if TP+FN > 0:
            recall = TP/(TP+FN)
        if TN+FP > 0:
            specificity = TN/(TN+FP)
        if TP+FP > 0:
            PPV = TP/(TP+FP)
        if TN+FN > 0:
            NPV = TN/(TN+FN)
        if TP+TN+FP+FN > 0:
            accuracy = (TP+TN)/(TP+TN+FP+FN)

        self.rateComplete = nComplete/len(self.data)
        self.rateCompletePos = nCompletePos/(nCompletePos+nIncompletePos)
        self.rateCompleteNeg = nCompleteNeg/(nCompleteNeg+nIncompleteNeg)
        print('%s+: completes: %i (%.2f%%), incompletes: %i (%.2f%%)'%(phenotype, nCompletePos, nCompletePos*100/(nCompletePos+nIncompletePos),nIncompletePos,nIncompletePos*100/(nCompletePos+nIncompletePos) ))
        print('%s-: completes: %i (%.2f%%), incompletes: %i (%.2f%%)'%(phenotype, nCompleteNeg, nCompleteNeg*100/(nCompleteNeg+nIncompleteNeg),nIncompleteNeg,nIncompleteNeg*100/(nCompleteNeg+nIncompleteNeg) ))
        print('%s: completes: %i (%.2f%%), incompletes: %i (%.2f%%)'%(phenotype, nComplete, nComplete*100/len(self.data),nIncomplete,nIncomplete*100/len(self.data) ))
        if nCompletePos == 0 or nCompleteNeg == 0:
            print('Warning: %s has insufficient complete genotypes, AEC modelling might be impossible'%phenotype)

        print('%s+: total: %i (%.2f%%) unique: %i (%.2f%%) conflicted: %i [%i] (%.2f%%)'%(phenotype, len(self.pos), len(self.pos)*100/len(self.phenotypes), len(self.poshash), len(self.poshash)*100/len(self.phenotypes),nConflictPos,len(conflictposhash),nConflictPos*100/len(self.pos) ))
        print('%s-: total: %i (%.2f%%) unique: %i (%.2f%%) conflicted: %i [%i] (%.2f%%)'%(phenotype, len(self.neg), len(self.neg)*100/len(self.phenotypes), len(self.neghash), len(self.neghash)*100/len(self.phenotypes),nConflictNeg,len(conflictneghash),nConflictNeg*100/len(self.neg) ))
        print('%s: conflicts: %i (%.2f%%), unique %i (%.2f%%) '%(phenotype, nConflict, nConflict*100/len(self.data), len(conflicthash), len(conflicthash)*100/len(self.allhash)))

        print('TP: %i TN: %i FP: %i FN: %i'%(TP,TN,FP,FN))
        print('recall: %.4f'%recall)
        print('specificity: %.4f'%specificity)
        print('PPV: %.4f'%PPV)
        print('NPV: %.4f'%NPV)
        print('Theoretical accuracy: %.4f'%accuracy)

    def split(self, train_rate, validate_rate, aec_mode = False):
        PYTHONHASHSEED = 0
        complete_pos_rate = len(self.completeposhash) / len(self.poshash)
        complete_neg_rate = len(self.completeneghash) / len(self.neghash)

        #Try to split unique genotypes according to requested rates, try again if the achieved rates too off
        adjustment = 0.0
        good_split = False
        attempt = 0
        max_attempts = 1000
        failed = False
        while not failed and not good_split and attempt < max_attempts:
            if complete_pos_rate >= train_rate+adjustment or complete_neg_rate >= train_rate+adjustment:
                aec_mode = False

            #Allocate unique genotypes (hashed) according to requested rates
            #if in AEC mode, try to allocate all complete genotypes for training set for the AE
            if aec_mode:
                poshash = list(self.poshash - self.completeposhash)
                neghash = list(self.neghash - self.completeneghash)
                train_pos_len = int(((train_rate+adjustment) - complete_pos_rate)*len(self.poshash))
                train_neg_len = int(((train_rate+adjustment) - complete_neg_rate)*len(self.neghash))
            else:
                poshash = list(self.poshash)
                neghash = list(self.neghash)
                train_pos_len = int((train_rate+adjustment)*len(self.poshash))
                train_neg_len = int((train_rate+adjustment)*len(self.neghash))

            random.shuffle(poshash)
            random.shuffle(neghash)

            validate_pos_len = int((validate_rate-adjustment)*len(self.poshash))
            validate_neg_len = int((validate_rate-adjustment)*len(self.neghash))
            test_pos_len = len(self.poshash) - train_pos_len - validate_pos_len
            test_neg_len = len(self.neghash) - train_neg_len - validate_neg_len

            if aec_mode:
                train_set = set(poshash[0:train_pos_len]) | self.completeposhash | set(neghash[0:train_neg_len]) | self.completeneghash
            else:
                train_set = set(poshash[0:train_pos_len]) | set(neghash[0:train_neg_len])
            validate_set = set(poshash[train_pos_len:train_pos_len+validate_pos_len]) | set(neghash[train_neg_len:train_neg_len+validate_neg_len])
            test_set = set(poshash[train_pos_len+validate_pos_len:]) | set(poshash[train_neg_len+validate_neg_len:])

            train_indices = []
            validate_indices = []
            test_indices = []

            nTrainPos = 0
            nTrainNeg = 0
            nValidatePos = 0
            nValidateNeg = 0
            nTestPos = 0
            nTestNeg = 0

            #Now allocate actual genotypes according to their allocated hash value, hopefully the rates will be somewhat similar
            #this might not be the case of there are very few unique genotypes, with very lopsided distributions
            for index in range(len(self.data)):
                h = hash(self.data[index].tostring())
                if h in train_set:
                    train_indices.append(index)
                    if self.phenotypes[index] == 0:
                        nTrainNeg += 1
                    else:
                        nTrainPos += 1                    
                elif h in validate_set:
                    validate_indices.append(index)
                    if self.phenotypes[index] == 0:
                        nValidateNeg += 1
                    else:
                        nValidatePos += 1                    
                else:
                    test_indices.append(index)
                    if self.phenotypes[index] == 0:
                        nTestNeg += 1
                    else:
                        nTestPos += 1

            good_split = True

            final_train_rate = len(train_indices) / self.data.shape[0]
            final_validate_rate = len(validate_indices) / self.data.shape[0]
            final_test_rate = len(test_indices) / self.data.shape[0]

            if final_train_rate == 0 or final_validate_rate == 0 or (final_test_rate == 0 and train_rate + validate_rate < 1 ):
                good_split = False

            if abs(final_train_rate - train_rate) > 0.25 or abs(final_validate_rate - validate_rate) > 0.15:
                good_split = False

            if nTrainPos == 0 or nTrainNeg == 0 or nValidatePos == 0 or nValidateNeg == 0:
                good_split = False

            if not good_split:
                if final_train_rate < train_rate:
                    adjustment += 0.05
                    if train_rate + adjustment > 1:
                        failed = True
                else:
                    adjustment -= 0.05
                    if train_rate + adjustment < 0:
                        failed = True

                print('Warning: Dataset split (%.2f/%.2f/%.2f) is bad, trying again'%( final_train_rate/(final_train_rate+final_validate_rate+final_test_rate),final_validate_rate/(final_train_rate+final_validate_rate+final_test_rate),final_test_rate/(final_train_rate+final_validate_rate+final_test_rate) ))
                attempt += 1

        if final_train_rate == 0 or final_validate_rate == 0 or (final_test_rate == 0 and train_rate + validate_rate < 1 ):
            return None, None, None
        elif not good_split:
            print('Warning: Final dataset split is bad, this might impact training!')

        print("Training set total: %d (%.2f), (+) %d (%.2f), (-) %d (%.2f)"%(len(train_indices), len(train_indices) / self.data.shape[0],nTrainPos,nTrainPos/(nTrainPos+nTrainNeg),nTrainNeg,nTrainNeg/(nTrainPos+nTrainNeg)))
        print("Validation set total: %d (%.2f), (+) %d (%.2f), (-) %d (%.2f)"%(len(validate_indices), len(validate_indices) / self.data.shape[0],nValidatePos,nValidatePos/(nValidatePos+nValidateNeg),nValidateNeg,nValidateNeg/(nValidatePos+nValidateNeg)))
        print("Testing set total: %d (%.2f), (+) %d (%.2f), (-) %d (%.2f)"%(len(test_indices),len(test_indices) / self.data.shape[0],nTestPos,nTestPos/(nTestPos+nTestNeg),nTestNeg,nTestNeg/(nTestPos+nTestNeg)))
        
        train_set = torch.utils.data.Subset(self,train_indices)
        validate_set = torch.utils.data.Subset(self,validate_indices)
        if len(test_indices) > 0:
            test_set = torch.utils.data.Subset(self,test_indices)
        else:
            test_set = None

        return train_set,validate_set,test_set

    def variants(self):
        return self.variants
    def cases(self):
        return self.cases
    
    def nVariants(self):
        return self.data.shape[1]
    def nCases(self):
        return self.data.shape[0]
    #def weights(self):
    #    return self.class_weight
    def pos_rate(self):
        return len(self.pos)/(len(self.neg)+len(self.pos))

    def pos_weight(self):
        return len(self.neg) / len(self.pos)

    def complete_rate(self):
        return self.rateComplete

    def complete_pos_rate(self):
        return self.rateCompletePos

    def complete_neg_rate(self):
        return self.rateCompleteNeg

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        #Return genotype, mutated genotype and phenotype for training, and id and genotype for prediction
        if self.train_mode:
            x = torch.FloatTensor(self.data[idx]).clone()
            y = x.clone()

            if self.mutationrate > 0 and round(np.random.rand()) == 0:
                nMut = int(round(self.mutationrate*len(y)))
                if nMut == 0:
                    nMut = 1
                y[np.random.choice(len(y), size=nMut )] = -1

            z = torch.FloatTensor([self.phenotypes[idx]])

            return [x, y, z]
        else:
            return self.cases[idx], torch.FloatTensor(self.data[idx])            

class PhenotyperMLP(nn.Module):
    def __init__(self,nIp,nhid,nOp):
        super(PhenotyperMLP, self).__init__()

        self.layer1 = nn.Linear(nIp,nhid)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(nhid,nhid)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(nhid,nhid)
        self.dropout3 = nn.Dropout(0.5)
        self.layer4 = nn.Linear(nhid,nOp)

    def mlp(self, x):
            x = F.relu(self.layer1(x))
            x = self.dropout1(x)
            x = F.relu(self.layer2(x))
            x = self.dropout2(x)
            x = F.relu(self.layer3(x))
            x = self.dropout3(x)
            x = torch.sigmoid(self.layer4(x))
            return x

    def forward(self, x_input):
        x_classified = self.mlp(x_input)
        return x_classified    

class PhenotyperAEC(nn.Module):
    def __init__(self,nIp,latent_dim,nOp):
        super(PhenotyperAEC, self).__init__()
        #Encode
        self.enc1 = nn.Conv1d(1,8,5, padding=2)
        self.enc2 = nn.Conv1d(8,16,5, padding=2)
        self.enc3 = nn.Conv1d(16,1,5, padding=2)
        self.enc4 = nn.Linear(nIp,latent_dim)
        
        #Decode
        self.dec1 = nn.Linear(latent_dim, nIp)
        self.dec2 = nn.ConvTranspose1d(1,16,5, padding=2)
        self.dec3 = nn.ConvTranspose1d(16,8,5, padding=2)
        self.dec4 = nn.ConvTranspose1d(8,1,5, padding=2)
        self.pool1 = nn.AvgPool1d(2, stride=2)
        
        #Classify
        self.layer1 = nn.Conv1d(1,8,5, padding=2)
        self.pool2 = nn.AvgPool1d(2, stride=2)
        self.layer2 = nn.Conv1d(8,16,5, padding=2)
        self.pool3 = nn.AvgPool1d(2, stride=2)
        self.layer3 = nn.Conv1d(16,1,5, padding=2)
        self.layer4 = nn.Linear(int(nIp/8), nOp)

    def encode(self, x_input):
        x = x_input.unsqueeze(1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x_latent = self.enc4(x)
        return x_latent
    
    def decode(self, x_latent):
        x = F.relu(self.dec1(x_latent))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x_imputed = self.dec4(x)
        return x_imputed.squeeze(1)

    def classify(self, x_imputed):
        x = self.pool1(F.relu(self.layer1(x_imputed.unsqueeze(1))))
        x = self.pool2(F.relu(self.layer2(x)))
        x = self.pool3(F.relu(self.layer3(x)))

        x_classified = torch.sigmoid(self.layer4(x.squeeze(1)))
        return x_classified

    def forward(self, x_input):
        #Autoencoder
        x_latent = self.encode(x_input)
        x_imputed = self.decode(x_latent)
        #Convolutional NN
        x_classified = self.classify(x_imputed)
        return x_imputed, x_classified

class DeepPhenoat:
    cases = {}

    def __init__(self, cases):
        self.cases = cases

    def Fbeta(self, beta, precision, recall):
        return(((1 + (beta**2)) * (precision * recall)) / ((beta**2) * precision + recall))

    def MLPtrainer(self, dataset, phenotype, min_accuracy):
        restarts = 0
        best_accuracy = 0.0
        best_loss = float('inf')

        while restarts <= min_restarts_mlp or (restarts < max_restarts and best_accuracy < min_accuracy):
            train_set, valid_set, test_set = dataset.split(data_rate_train,data_rate_validate, False)

            if train_set == None:
                print('Achtung: Smart MLP dataset split for %s failed, resorting to simple random splitting!'%phenotype)

                nTrain_set = int(dataset.nCases()*data_rate_train)
                nValid_set = int(dataset.nCases()*data_rate_validate)
                nTest_set = dataset.nCases() - nTrain_set - nValid_set
                if data_rate_train+data_rate_validate < 1:
                    train_set, valid_set, test_set = random_split(dataset,[nTrain_set,nValid_set,nTest_set])
                else:
                    train_set, valid_set = random_split(dataset,[nTrain_set,nValid_set])
                    test_set = valid_set
                print('Training set: %d (%.2f)'%(len(train_set), len(train_set)/dataset.nCases()) )
                print('Validation set: %d (%.2f)'%(len(valid_set), len(valid_set)/dataset.nCases()) )
                print('Test set: %d (%.2f)'%(len(test_set), len(test_set)/dataset.nCases()) )

            if test_set == None:
                test_set = valid_set

            train_loader = DataLoader(train_set,batch_size=1,shuffle=True)
            valid_loader = DataLoader(valid_set,batch_size=10,shuffle=True)
            test_loader  = DataLoader(test_set,batch_size=10,shuffle=True)


            nhid = int(round(dataset.nVariants()*hid_factor))
            if nhid < min_hid:
                nhid = min_hid

            out = open(workdir+'AI.params.%s.mlp'%(phenotype), "wt")
            out.write('nIp\thid\tnOp\n')
            out.write('%i\t%i\t%i\n'%(dataset.nVariants(),nhid,1))
            out.close()

            print('hid dim: %i'%nhid)
            mlp = PhenotyperMLP(nIp=dataset.nVariants(),nhid=nhid,nOp=1)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(mlp.parameters(),lr=1e-3)

            trLoss = []
            vlLoss = []
            epochs = 1000
            #retries = 75
            attempt = 0
            #best_mlp = mlp.state_dict().copy()
            TP = 0
            TN = 0
            FP = 0
            FN = 0

            print('Training...')
            t_start = datetime.timestamp(datetime.now())
            for epoch in range(epochs):
                #Train
                epLoss = 0
                counter = 0
                mlp.train()
                for x_raw, _, x_phenotype in train_loader:
                    #weight = np.full_like(x_phenotype, dataset.pos_weight())
                    #weight[x_phenotype == 0.0] = 1.0/dataset.pos_weight()

                    x_classified = mlp(x_raw)
                    loss = criterion(x_classified,x_phenotype)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epLoss += loss.item() * x_raw.size(0)
                    counter += x_raw.size(0)
                trLoss.append(epLoss/counter)
                #Validate
                epLoss = 0
                counter = 0
                mlp.eval()
                for x_raw, _, x_phenotype in valid_loader:
                    #weight = np.full_like(x_phenotype, dataset.pos_weight())
                    #weight[x_phenotype == 0.0] = 1.0/dataset.pos_weight()

                    with torch.no_grad():
                        x_classified = mlp(x_raw)
                        loss = criterion(x_classified,x_phenotype)
                        epLoss += loss.item() * x_raw.size(0)
                        counter += x_raw.size(0)

                    x_classified = x_classified.squeeze(1).detach().numpy().round()
                    x_phenotype = x_phenotype.squeeze(1).detach().numpy()

                    for index in range(len(x_phenotype)):
                        if x_phenotype[index] == 1:
                            if x_classified[index] == 1:
                                TP = TP + 1
                            elif x_classified[index] == 0:
                                FN = FN + 1
                        elif x_phenotype[index] == 0:
                            if x_classified[index] == 0:
                                TN = TN + 1
                            elif x_classified[index] == 1:
                                FP = FP + 1

                vlLoss.append(epLoss/counter)
                accuracy = ((TP/(TP+FN))+(TN/(TN+FP)))/2
                #accuracy = (TP+TN)/(TP+TN+FP+FN)

                if epoch == 0:
                    initial_accuracy = accuracy

                if (TP > 0 and TN > 0) and (accuracy >= best_accuracy or (round(accuracy,4) == round(best_accuracy,4) and vlLoss[-1] < best_loss)):
                    best_loss = vlLoss[-1]
                    best_accuracy = accuracy
                    best_mlp = mlp.state_dict().copy()
                    if round(best_accuracy,2) >= round(min_accuracy,2):
                        torch.save(mlp.state_dict(), workdir+'AI.training.%s.round(%i).epoch(%i).loss(%0.4f).acc(%.4f).mlp'%(phenotype,restarts,epoch,vlLoss[-1],accuracy) )
                    attempt = 0
                else:
                    attempt = attempt + 1

                if attempt > retries_mlp:
                    print('Epoch: %03d, max retries reached, stopping early'%(epoch))
                    break

                if epoch % 5 == 0:
                    t_end = datetime.timestamp(datetime.now())
                    est = (t_end - t_start)/ (epoch+1)
                    hours = 0
                    mins = 0
                    secs = 0

                    if est > (60*60):
                        hours = int(est / (60*60))
                        est = est - (hours * 60*60)
                    if est > (60):
                        mins = int(est / 60)
                        est = est - (mins * 60)

                    secs = int(est)
                    print('Epoch: %03d, Attempt: %03d, Epoch length: %dh:%dm:%d:s, Tr.Loss: %.8f, Vl.Loss: %.8f, Vl.Acc: %.8f'%(epoch,attempt,hours,mins,secs,trLoss[-1],vlLoss[-1], accuracy))

                if epoch >= 40 and accuracy < saddle_accuracy and best_accuracy - initial_accuracy < 0.01:
                    print('Warning: %s model training seems to be stuck inside saddle point, trying again!'%(phenotype))
                    break

            if restarts < max_restarts and best_accuracy < min_accuracy:
                print('Warning: %s achieved accuracy of %.2f is below threhold of %0.2f, trying again!'%(phenotype, best_accuracy,min_accuracy))
            else:
                print('MLP model: %s, round: %i of %i, best loss: %.8f, best accuracy: %.08f'%(phenotype, restarts+1, min_restarts_mlp+1, best_loss,best_accuracy))
            restarts += 1

        out = open(workdir+'AI.training.loss.%s.mlp'%phenotype, "wt")
        out.write('training\tvalidation\n')
        for index in range(len(trLoss)):
            out.write('%f\t%f\n'%(trLoss[index],vlLoss[index]))
        out.close()

        if best_accuracy == 0:
            print('Achtung: Training MLP model for phenotype %s failed!'%(phenotype))
            return(min_accuracy)

        torch.save(best_mlp, workdir+'AI.model.%s.mlp'%(phenotype) )
        mlp.load_state_dict(best_mlp)
        mlp.eval()
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for x_raw, _, x_phenotype in test_loader:
            with torch.no_grad():
                x_classified = mlp(x_raw)

            x_classified = x_classified.squeeze(1).detach().numpy().round()
            x_phenotype = x_phenotype.squeeze(1).detach().numpy()

            for index in range(len(x_phenotype)):
                if x_phenotype[index] == 1:
                    if x_classified[index] == 1:
                        TP += 1
                    elif x_classified[index] == 0:
                        FN += 1
                elif x_phenotype[index] == 0:
                    if x_classified[index] == 0:
                        TN += 1
                    elif x_classified[index] == 1:
                        FP += 1

        recall = 0
        specificity = 0 
        PPV = 0
        NPV = 0
        accuracy = 0
        fscore = 0

        if TP+FN > 0:
            recall = TP/(TP+FN)
        if TN+FP > 0:
            specificity = TN/(TN+FP)
        if TP+FP > 0:
            PPV = TP/(TP+FP)
        if TN+FN > 0:
            NPV = TN/(TN+FN)
        if TP+TN+FP+FN > 0:
            accuracy = (TP+TN)/(TP+TN+FP+FN)

        rate = dataset.pos_rate()
        if PPV > 0 or recall > 0:
            fscore = self.Fbeta(1.0, PPV, recall)

        print('set with ' + str(len(test_set)) + ' cases and ' + str(dataset.nVariants()) + ' variants' )
        print('TP: %i TN: %i FP: %i FN: %i'%(TP,TN,FP,FN))
        print('recall: %.4f'%recall)
        print('specificity: %.4f'%specificity)
        print('PPV: %.4f'%PPV)
        print('NPV: %.4f'%NPV)
        print('achieved accuracy: %.4f'%accuracy)

        if TP == 0:
            print('Achtung: %s model has a recall rate of 0 and is therefore invalid.')
        if TN == 0:
            print('Achtung: %s model has a specificity rate of 0 and is therefore invalid.')

        out = open(workdir+'AI.evaluation.training.mlp', "a")
        out.write('%s\t%i\t%i\t%i\t%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%i\t%i\t%.8f\n'%(phenotype,TP,TN,FP,FN,recall,specificity,PPV,NPV,accuracy,fscore,dataset.nCases(),dataset.nVariants(),rate))
        out.close()

        out = open(workdir+'AI.evaluation.%s.mlp'%phenotype, "w")
        out.write('phenotype\tTP\tTN\tFP\tFN\trecall\tspecificity\tPPV\tNPV\taccuracy\tFscore\tcases\tvariants\t(+)rate\n')
        out.write('%s\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%i\t%i\t%f\n'%(phenotype,TP,TN,FP,FN,recall,specificity,PPV,NPV,accuracy,fscore,dataset.nCases(),dataset.nVariants(),rate))
        out.close()

        print('Trained MLP model for phenotype: %s, accuracy: %.6f, ids: %i, variants: %i, (+)rate: %.6f'%(phenotype, accuracy,dataset.nCases(),dataset.nVariants(),rate))
        return(best_accuracy)

    def AECtrainer(self, dataset, phenotype, min_accuracy):
        abort = False
        restarts = 0
        best_accuracy = 0.0
        best_loss = float('inf')

        while not abort and restarts < max_restarts and round(best_accuracy,2) + 0.01 < round(min_accuracy,2):
            train_set, valid_set, test_set = dataset.split(data_rate_train,data_rate_validate, True)

            if train_set == None:
                print('Achtung: Smart AEC dataset split for %s failed, resorting to simple random splitting!'%phenotype)
                nTrain_set = int(dataset.nCases()*data_rate_train)
                nValid_set = int(dataset.nCases()*data_rate_validate)
                nTest_set = dataset.nCases() - nTrain_set - nValid_set
                if data_rate_train+data_rate_validate < 1:
                    train_set, valid_set, test_set = random_split(dataset,[nTrain_set,nValid_set,nTest_set])
                else:
                    train_set, valid_set = random_split(dataset,[nTrain_set,nValid_set])
                    test_set = valid_set
                print('Training set: %d (%.2f)'%(len(train_set), len(train_set)/dataset.nCases()) )
                print('Validation set: %d (%.2f)'%(len(valid_set), len(valid_set)/dataset.nCases()) )
                print('Test set: %d (%.2f)'%(len(test_set), len(test_set)/dataset.nCases()) )

            if test_set == None:
                test_set = valid_set

            latent_dim = int(round(dataset.nVariants()*dim_factor))
            if latent_dim < min_dim:
                latent_dim = min_dim

            out = open(workdir+'AI.params.%s.aec'%(phenotype), "wt")
            out.write('nIp\tlatent_dim\tnOp\n')
            out.write('%i\t%i\t%i\n'%(dataset.nVariants(),latent_dim,1))
            out.close()

            print('Latent dim: %i'%latent_dim)
            model = PhenotyperAEC(nIp=dataset.nVariants(),latent_dim=latent_dim,nOp=1)
            criterion1 = nn.MSELoss()
            criterion2 = nn.BCELoss()

            optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

            print('Training...')
            trLoss = []
            vlLoss = []
            if dataset.complete_pos_rate() > 0.7 and dataset.complete_neg_rate() > 0.7 and dataset.nCases() > 50000:
                inflation_factor_train_phase1 = 2
                inflation_factor_train_phase2 = 1
            elif dataset.complete_pos_rate() > 0.6 and dataset.complete_neg_rate() > 0.6 and dataset.nCases() > 50000:
                inflation_factor_train_phase1 = 2
                inflation_factor_train_phase2 = 2
            elif dataset.complete_pos_rate() > 0.4 and dataset.complete_neg_rate() > 0.4 and dataset.nCases() > 50000:
                inflation_factor_train_phase1 = 3
                inflation_factor_train_phase2 = 2
            else:
                inflation_factor_train_phase1 = 10
                inflation_factor_train_phase2 = 5

            inflation_factor_validate = 2
            epochs = 1000
            #retries = 5
            attempt = 0

            print('inflation factor train phase1: %i, train phase2: %i, validate: %i'%(inflation_factor_train_phase1,inflation_factor_train_phase2,inflation_factor_validate))

            t_start = datetime.timestamp(datetime.now())
            for epoch in range(epochs):
                #Train phase1
                train_loader = DataLoader(train_set,batch_size=1,shuffle=True)
                epLoss = 0
                counter = 0
                model.train()
                for _ in range(0,inflation_factor_train_phase1):
                    for x_raw, x_mutated, x_phenotype in train_loader:
                        if int(x_raw.min()) < 0:
                            continue

                        #weight = np.full_like(x_phenotype, dataset.pos_weight())
                        #weight[x_phenotype == 0.0] = 1.0/dataset.pos_weight()

                        x_imputed,x_classified = model(x_mutated)
                        if x_mutated.min() < 0:
                            loss1 = criterion1(x_imputed[x_mutated==-1],x_raw[x_mutated==-1])
                            loss2 = criterion2(x_classified,x_phenotype)
                            loss = loss1*0.75 + loss2*0.25
                        else:
                            loss1 = criterion1(x_imputed,x_raw)
                            loss2 = criterion2(x_classified,x_phenotype)
                            loss = loss1*0.5 + loss2*0.5

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epLoss += loss.item() * x_raw.size(0)
                        counter += x_raw.size(0)

                #Train phase2
                train_loader = DataLoader(train_set,batch_size=2,shuffle=True)
                for _ in range(0,inflation_factor_train_phase2):
                    for x_raw, x_mutated, x_phenotype in train_loader:
                        x_imputed,x_classified = model(x_mutated)

                        loss1 = criterion1(x_imputed[x_raw!=-1],x_raw[x_raw!=-1])
                        loss2 = criterion2(x_classified,x_phenotype)
                        loss = loss1*0.5 + loss2*0.5

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epLoss += loss.item() * x_raw.size(0)
                        counter += x_raw.size(0)

                trLoss.append(epLoss/counter)

                #Validate
                epLoss = 0
                counter = 0
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                model.eval()
                valid_loader = DataLoader(valid_set,batch_size=10,shuffle=True)
                for _ in range(0,inflation_factor_validate):
                    for x_raw, x_mutated, x_phenotype in valid_loader:
                        with torch.no_grad():
                            x_imputed,x_classified = model(x_mutated)
                            loss1 = criterion1(x_imputed[x_raw!=-1],x_raw[x_raw!=-1])
                            loss2 = criterion2(x_classified,x_phenotype)
                            loss = loss1*0.5 + loss2*0.5

                        x_classified = x_classified.squeeze(1).detach().numpy().round()
                        x_phenotype = x_phenotype.squeeze(1).detach().numpy()
                        epLoss += loss.item() * x_raw.size(0)
                        counter += x_raw.size(0)

                        for index in range(len(x_phenotype)):
                            if x_phenotype[index] == 1:
                                if x_classified[index] == 1:
                                    TP = TP + 1
                                elif x_classified[index] == 0:
                                    FN = FN + 1
                            elif x_phenotype[index] == 0:
                                if x_classified[index] == 0:
                                    TN = TN + 1
                                elif x_classified[index] == 1:
                                    FP = FP + 1

                #accuracy = (TP+TN)/(TP+TN+FP+FN)
                accuracy = ((TP/(TP+FN))+(TN/(TN+FP)))/2
                vlLoss.append(epLoss/counter)

                if epoch == 0:
                    initial_accuracy = accuracy

                if (TP > 0 and TN > 0) and (accuracy > best_accuracy or (round(accuracy,4) == round(best_accuracy,4) and vlLoss[-1] < best_loss)):
                    best_loss = vlLoss[-1]
                    best_accuracy = accuracy

                    best_model = model.state_dict().copy()
                    if round(best_accuracy,2) >= round(min_accuracy,2):
                        torch.save(model.state_dict(), workdir+'AI.training.%s.epoch(%i).loss(%0.4f).acc(%0.4f).aec'%(phenotype,epoch,vlLoss[-1],accuracy) )

                    attempt = 0
                else:
                    attempt = attempt + 1
                
                    if attempt > retries_aec:
                        print('Epoch: %03d, max retries reached, stopping'%(epoch))
                        break

                if epoch >= 4 and accuracy < saddle_accuracy and best_accuracy - initial_accuracy < 0.01:
                    print('Warning: %s model training seems to be stuck inside saddle point, trying again!'%(phenotype))
                    break

                t_end = datetime.timestamp(datetime.now())
                est = (t_end - t_start)/ (epoch + 1)
                hours = 0
                mins = 0
                secs = 0

                if est > (60*60):
                    hours = int(est / (60*60))
                    est = est - (hours * 60*60)
                if est > (60):
                    mins = int(est / 60)
                    est = est - (mins * 60)

                secs = int(est)
                print('Epoch: %03d, Attempt: %03d, Epoch length: %dh:%dm:%d:s, Tr.Loss: %.8f, Vl.Loss: %.8f, Vl.Acc: %.8f'%(epoch,attempt,hours,mins,secs,trLoss[-1],vlLoss[-1], accuracy))
                if (limitHours and datetime.timestamp(datetime.now()) - t_start) / (60*60) > maxTrainingHours:
                    print('Training %s has surpassed %i hours, stopping early'%(phenotype,maxTrainingHours))
                    abort = True
                    break

            print('AEC model: %s, round: %i, best loss: %.8f, best accuracy: %.8f'%(phenotype, restarts, best_loss, best_accuracy))
            if not abort and restarts < max_restarts and round(best_accuracy,2) + 0.01 < round(min_accuracy,2):
                print('Warning: %s achieved accuracy of %0.2f is below threshold of %0.2f, trying again!'%(phenotype, best_accuracy, min_accuracy))
            restarts += 1

        out = open(workdir+'AI.training.loss.%s.aec'%(phenotype), "wt")
        out.write('training\tvalidation\n')
        for index in range(len(trLoss)):
            out.write('%f\t%f\n'%(trLoss[index],vlLoss[index]))
        out.close()

        if best_accuracy == 0:
            print('Achtung: Training AEC model for phenotype %s failed!'%(phenotype))
            return(min_accuracy)

        model.load_state_dict(best_model)
        torch.save(model.state_dict(), workdir+'AI.model.%s.aec'%(phenotype) )
        model.eval()
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        test_loader = DataLoader(test_set,batch_size=10,shuffle=False)
        for (x_raw, _, x_phenotype) in test_loader:
            with torch.no_grad():
                _, x_classified=model(x_raw)

            x_classified = x_classified.squeeze(1).detach().numpy().round()
            x_phenotype = x_phenotype.squeeze(1).detach().numpy()

            for index in range(len(x_phenotype)):
                if x_phenotype[index] == 1:
                    if x_classified[index] == 1:
                        TP = TP + 1
                    elif x_classified[index] == 0:
                        FN = FN + 1
                elif x_phenotype[index] == 0:
                    if x_classified[index] == 0:
                        TN = TN + 1
                    elif x_classified[index] == 1:
                        FP = FP + 1

        recall = 0
        specificity = 0 
        PPV = 0
        NPV = 0
        accuracy = 0
        fscore = 0

        if TP+FN > 0:
            recall = TP/(TP+FN)
        if TN+FP > 0:
            specificity = TN/(TN+FP)
        if TP+FP > 0:
            PPV = TP/(TP+FP)
        if TN+FN > 0:
            NPV = TN/(TN+FN)
        if TP+TN+FP+FN > 0:
            accuracy = (TP+TN)/(TP+TN+FP+FN)

        rate = dataset.pos_rate()

        if PPV > 0 or recall > 0:
            fscore = self.Fbeta(1.0, PPV, recall)

        print('Testset with ' + str(len(test_set)) + ' cases and ' +str(dataset.nVariants())+ ' variants' )
        print('TP: %i TN: %i FP: %i FN: %i'%(TP,TN,FP,FN))
        print('recall: %.4f'%recall)
        print('specificity: %.4f'%specificity)
        print('PPV: %.4f'%PPV)
        print('NPV: %.4f'%NPV)
        print('achieved accuracy: %.4f'%accuracy)

        if TP == 0:
            print('Achtung: %s AEC model has a recall rate of 0 and is therefore invalid.'%phenotype)
        if TN == 0:
            print('Achtung: %s AEC model has a specificity rate of 0 and is therefore invalid.'%phenotype)

        out = open(workdir+'AI.evaluation.training.aec', "a")
        out.write('%s\t%i\t%i\t%i\t%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%i\t%i\t%.8f\n'%(phenotype,TP,TN,FP,FN,recall,specificity,PPV,NPV,accuracy,fscore,dataset.nCases(),dataset.nVariants(),rate))
        out.close()

        out = open(workdir+'AI.evaluation.%s.aec'%phenotype, "w")
        out.write('phenotype\tTP\tTN\tFP\tFN\trecall\tspecificity\tPPV\tNPV\taccuracy\tFscore\tcases\tvariants\t(+)rate\n')
        out.write('%s\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%i\t%i\t%f\n'%(phenotype,TP,TN,FP,FN,recall,specificity,PPV,NPV,accuracy,fscore,dataset.nCases(),dataset.nVariants(),rate))
        out.close()

        print('Trained AEC model for phenotype: %s, accuracy: %.6f, ids: %i, variants: %i, (+)rate: %.6f'%(phenotype, accuracy,dataset.nCases(),dataset.nVariants(),rate))
        return(best_accuracy)

    def generate(self, phenotype, model, dataset, file,model_type):
        batch_size = 128
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)

        TP = 0
        FN = 0
        FP = 0
        TN = 0
        predictions = 0
        progress = Progress(dataset.nCases())
        for x_id, x_genotype in data_loader:
            if model_type == 'mlp':
                x_class = model(x_genotype)
            else:
                _, x_class = model(x_genotype)
            x_class = x_class.squeeze(1).detach().numpy().round()
            for index in range(len(x_id)):
                progress.step()
                file.write('%s\t%i\n'%(x_id[index],x_class[index]))
                predictions += 1
                if x_id[index] in self.cases:
                    case = self.cases[x_id[index]]
                    if phenotype in case.phenotypes:
                        if case.phenotypes[phenotype] == 0:
                            if x_class[index] == 0:
                                TN += 1
                            elif x_class[index] == 1:
                                FP += 1
                        elif case.phenotypes[phenotype] == 1:
                            if x_class[index] == 0:
                                FN += 1
                            elif x_class[index] == 1:
                                TP += 1

        file.close()
        print('')
        out = open(workdir+'AI.evaluation.prediction.%s'%model_type, "a")
        if TP+TN+FP+FN > 0:
            recall = 0
            specificity = 0 
            PPV = 0
            NPV = 0
            accuracy = 0
            fscore = 0
            if TP+FN > 0:
                recall = TP/(TP+FN)
            if TN+FP > 0:
                specificity = TN/(TN+FP)
            if TP+FP > 0:
                PPV = TP/(TP+FP)
            if TN+FN > 0:
                NPV = TN/(TN+FN)
            if TP+TN+FP+FN > 0:
                accuracy = (TP+TN)/(TP+TN+FP+FN)

            if PPV > 0 or recall > 0:
                fscore = self.Fbeta(1.0, PPV, recall)

            print('TP: %i TN: %i FP: %i FN: %i'%(TP,TN,FP,FN))
            print('recall: %.4f'%recall)
            print('specificity: %.4f'%specificity)
            print('PPV: %.4f'%PPV)
            print('NPV: %.4f'%NPV)
            print('prediction accuracy: %.4f, fscore: %.4f'%(accuracy,fscore))
            out.write('%s\t%i\t%i\t%i\t%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'%(phenotype,TP,TN,FP,FN,recall,specificity,PPV,NPV,accuracy,fscore))
        else:
            out.write('%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n'%phenotype)
        out.close()

        return predictions

    def predict(self, file):
        print('Initiating AI phenotype prediction')

        if MLP:
            out = open(workdir+'AI.evaluation.prediction.mlp', "w")
            out.write('phenotype\tTP\tTN\tFP\tFN\trecall\tspecificity\tPPV\tNPV\taccuracy\tFscore\n')
            out.close()

        if AEC:
            out = open(workdir+'AI.evaluation.prediction.aec', "w")
            out.write('phenotype\tTP\tTN\tFP\tFN\trecall\tspecificity\tPPV\tNPV\taccuracy\tFscore\n')
            out.close()

        file = open(file, "r")

        for line in file:
            if line.startswith('#'):
                continue

            items = line.rstrip().split('\t')

            loci = items[0].split('/')
            system = items[1]
            phenotypes = items[2]

            print('Processing genotypes at loci ' + items[0] + ' for phenotypes: ' + phenotypes)

            for phenotype in phenotypes.split('/'):
                if phenotype in skipped:
                    print('Skipping phenotype %s'%phenotype )
                    continue

                if MLP: #and phenotype not in skip_mlp:
                    variants = workdir+'AI.variants.%s.mlp'%phenotype
                    predictions = workdir + 'AI.predictions.%s.mlp'%phenotype
                    params = workdir + 'AI.params.%s.mlp'%phenotype
                    if not exists(variants) or not exists(params) or not exists(workdir+'AI.model.%s.mlp'%phenotype):
                        print('Error: %s lacks necessary MLP model files, cannot perform prediction!'%phenotype)
                    elif resume and exists(predictions):
                        print('%s MLP predictions already exist, skipping!'%phenotype)
                    else:
                        params = open(params)
                        params.readline()
                        params = params.readline().rstrip().split('\t')
                        nIp = int(params[0])
                        nhid = int(params[1])
                        nOp = int(params[2])
                        model = PhenotyperMLP(nIp=nIp,nhid=nhid,nOp=1)
                        model.load_state_dict(torch.load(workdir+'AI.model.%s.mlp'%phenotype))
                        model.eval()
                        dataset = AIDataset(phenotype, loci, None, None,0,1,variants)
                        predictions = open(predictions,'w')
                        predictions.write('ID\t%s\n'%phenotype)
                        print('Predicting MLP phenotypes for %s...'%phenotype)
                        predictions = self.generate(phenotype, model, dataset, predictions,'mlp')
                        print('Generated %i predictions using MLP model: %s'%(predictions,phenotype))

                if AEC:# and phenotype not in skip_aec:
                    variants = workdir+'AI.variants.%s.aec'%phenotype
                    predictions = workdir + 'AI.predictions.%s.aec'%phenotype
                    params = workdir + 'AI.params.%s.aec'%phenotype
                    if not exists(variants) or not exists(params) or not exists(workdir+'AI.model.%s.aec'%phenotype):
                        print('Error: %s lacks necessary AEC model files, cannot perform prediction!'%phenotype)
                    elif resume and exists(predictions):
                        print('%s AEC predictions already exist, skipping!'%phenotype)
                    else:
                        params = open(params)
                        params.readline()
                        params = params.readline().rstrip().split('\t')
                        nIp = int(params[0])
                        latent_dim = int(params[1])
                        nOp = int(params[2])
                        model = PhenotyperAEC(nIp=nIp,latent_dim=latent_dim,nOp=nOp)
                        model.load_state_dict(torch.load(workdir+'AI.model.%s.aec'%phenotype))
                        model.eval()
                        dataset = AIDataset(phenotype, loci,None, None,0,1,variants)
                        predictions = open(predictions,'w')
                        predictions.write('ID\t%s\n'%phenotype)
                        print('Predicting AEC phenotypes for %s...'%phenotype)
                        predictions = self.generate(phenotype, model, dataset, predictions,'aec')
                        print('Generated %i predictions using AEC model: %s'%(predictions,phenotype))

        file.close()
        print('Finished AI phenotype prediction.')

    def train(self, file, min_accuracy):
        print('Initiating machine learning:')
        file = open(file, "r")
        self.min_accuracy = min_accuracy

        skip_mlp = []
        skip_aec = []

        if not resume:
            if MLP:
                out = open(workdir+'AI.evaluation.training.mlp', "w")
                out.write('phenotype\tTP\tTN\tFP\tFN\trecall\tspecificity\tPPV\tNPV\taccuracy\tFscore\tcases\tvariants\t(+)rate\n')
                out.close()

            if AEC:
                out = open(workdir+'AI.evaluation.training.aec', "w")
                out.write('phenotype\tTP\tTN\tFP\tFN\trecall\tspecificity\tPPV\tNPV\taccuracy\tFscore\tcases\tvariants\t(+)rate\n')
                out.close()
        else:
            evaluation = workdir+'AI.evaluation.training.mlp'
            if exists(evaluation):
                evaluation = open(evaluation, "r")
                header = True
                for line in evaluation:
                    if header:
                        header = False
                        continue
                    items = line.rstrip().split('\t')
                    skip_mlp.append(items[0])
                evaluation.close()
            if MLP:
                print('MLP resuming after: ')
            evaluation = workdir+'AI.evaluation.training.aec'
            if exists(evaluation):
                evaluation = open(evaluation, "r")
                header = True
                for line in evaluation:
                    if header:
                        header = False
                        continue
                    items = line.rstrip().split('\t')
                    skip_aec.append(items[0])
                evaluation.close()

        for line in file:
            if line.startswith('#'):
                continue

            items = line.rstrip().split('\t')

            loci = items[0].split('/')
            system = items[1]
            phenotypes = items[2]

            print('Processing genotypes at loci ' + items[0] + ' for phenotypes: ' + phenotypes)

            for phenotype in phenotypes.split('/'):
                if phenotype in skipped:
                    print('Skipping phenotype %s'%phenotype )
                    continue

                pos = []
                neg = []
                for id in self.cases:
                    case = self.cases[id]
                    if phenotype in case.phenotypes and phenotype not in case.conflicts:
                        if case.phenotypes[phenotype] == 0:
                            neg.append(case.id)
                        elif case.phenotypes[phenotype] == 1:
                            pos.append(case.id)
                        else:
                            print('Error: Invalid ' + phenotype + ' phenotype: ' + case.phenotypes[phenotype] )
                            exit(1)

                print('')
                if len(pos) + len(neg) == 0:
                    print('Warning: phenotype ' + phenotype + ' has no cases and is therefore skipped for training!' )
                    if MLP and phenotype not in skip_mlp:
                        out = open(workdir+'AI.evaluation.training.mlp', "a")
                        out.write('%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t0\tNA\n'%phenotype)
                        out.close()
    
                    if AEC and phenotype not in skip_aec:
                        out = open(workdir+'AI.evaluation.training.aec', "a")
                        out.write('%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t0\tNA\n'%phenotype)
                        out.close()
                    continue

                rate = len(pos)/(len(neg)+len(pos))

                if rate == 0 or rate == 1:
                    print('Warning: phenotype ' + phenotype + ' has a (+)rate of ' + str(round(rate,2)) + ' and is therefore skipped for training!' )
                    if MLP and phenotype not in skip_mlp:
                        out = open(workdir+'AI.evaluation.training.mlp', "a")
                        out.write('%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t0\t%.1f\n'%(phenotype,rate))
                        out.close()
    
                    if AEC and phenotype not in skip_aec:
                        out = open(workdir+'AI.evaluation.training.aec', "a")
                        out.write('%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t0\t%.1f\n'%(phenotype,rate))
                        out.close()
                    continue

                goal_accuracy = self.min_accuracy
                if MLP and phenotype not in skip_mlp:
                    model = workdir + 'AI.model.%s.mlp'%phenotype
                    if exists(model):
                        print('Trained %s MLP model already exists, skipping!'%phenotype)
                    else:
                        training_variants = workdir + 'AI.training.variants.%s.mlp'%phenotype
                        if exists(training_variants):
                            print('Training %s MLP model on specified variants'%phenotype)
                        else:
                            print('Training %s MLP model on all variants'%phenotype)
                            training_variants = None
                        dataset = AIDataset(phenotype, loci, pos, neg, 0.3, 1.0,training_variants)
                        if len(dataset.variants) == 0 or len(dataset.cases) == 0:
                            print('Warning: %s has insufficient data for MLP training, skipping'%phenotype)
                        else:
                            out = open(workdir+'AI.variants.%s.mlp'%phenotype, "w")
                            out.write('chr\tpos\tallele1\tallele2\tmaf\tna\thom.a1\thet\thom.a2\tmissing\n')
                            for variant in dataset.variants:
                                out.write('%s\t%i\t%s\t%s\t%f\t%f\t%i\t%i\t%i\t%i\n'%(variant.chr,variant.pos,variant.allele1,variant.allele2,variant.maf,variant.na,variant.homREF,variant.het,variant.homALT,variant.missing))
                            out.close()
                            print('\nTraining MLP model for phenotype %s'%(phenotype))
                            goal_accuracy = self.MLPtrainer(dataset,phenotype,goal_accuracy)
                            print('')
                if AEC and phenotype not in skip_aec:
                    model = workdir + 'AI.model.%s.aec'%phenotype
                    if exists(model):
                        print('Trained %s AEC model already exists, skipping!'%phenotype)
                    else:
                        training_variants = workdir + 'AI.training.variants.%s.aec'%phenotype
                        if exists(training_variants):
                            print('Training %s AEC model on specified variants'%phenotype)
                        else:
                            print('Training %s AEC model on all variants'%phenotype)
                            training_variants = None

                        dataset = AIDataset(phenotype, loci, pos, neg, 0.3, 0.5,training_variants)
                        if len(dataset.variants) == 0 or len(dataset.cases) == 0:
                            print('Warning: %s has insufficient data for AEC training, skipping'%phenotype)
                        else:
                            out = open(workdir+'AI.variants.%s.aec'%phenotype, "w")
                            out.write('chr\tpos\tallele1\tallele2\tmaf\tna\thom.a1\thet\thom.a2\tmissing\n')
                            for variant in dataset.variants:
                                out.write('%s\t%i\t%s\t%s\t%f\t%f\t%i\t%i\t%i\t%i\n'%(variant.chr,variant.pos,variant.allele1,variant.allele2,variant.maf,variant.na,variant.homREF,variant.het,variant.homALT,variant.missing))
                            out.close()
                            print('\nTraining AEC model for phenotype %s'%(phenotype))
                            self.AECtrainer(dataset,phenotype,goal_accuracy)
                            print('')

        file.close()
        print('Finished machine learning.')

print('bloodAI v%.1f\n'%version)

phenotyper = Phenotyper()
phenotyper.load(workdir+phenotypes_file)

print('cpus: %i'%os.cpu_count())
print('cudas: %i'%torch.cuda.device_count())
print('threads: %i'%torch.get_num_threads())

deeptrainer = DeepPhenoat(phenotyper.cases)

if perform_training:
    deeptrainer.train(workdir + phenotypes_info_file, min_accuracy)
if perform_prediction:
    deeptrainer.predict(workdir + phenotypes_info_file)

print("Done!")
