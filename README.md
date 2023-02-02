# Orca leukemia
This repository contains code for scoring leukemia associated structural variants described in the Structural Variation Cooperates with Permissive Chromatin to Control Enhancer Hijacking-Mediated Oncogenic Transcription manuscript. The code can be used to score structural any variant-gene pair, where the structural variant can be a translocation, inversion, duplication, or deletion and the gene should be not further than 500Kb from the breakpoint. The scoring could be done in one of the following tissues: T-ALL_GSE134761, THP1, GM12878, NALM6, Non-ETP_GSE146901, ETP_GSE146901, K562, KBM7.


## Installation
To run the Orca leukemia scoring first Orca needs to be installed, more detailed installation steps are available here: https://github.com/jzhoulab/orca. Breifely,
1. Copy the Orca repository
```
git clone https://github.com/jzhoulab/orca.git
cd orca 
```

2. Set up the following conda environment
```
conda env create -f orca_env.yml
```

3. Install pytorch (>=1.7.0) according to [here](https://pytorch.org/get-started/locally/)
4. Install Selene
```
git clone https://github.com/kathyxchen/selene.git
cd selene
git checkout custom_target_support
python setup.py build_ext --inplace
python setup.py install
```

5. Download the resource package required to run Orca Leukimia
```
wget https://zenodo.org/record/7600893/files/orca_leukemia.tar.gz
tar xf orca_leukemia.tar.gz
```
8. Next download additional files required for Orca Leukemia Scoring 
```
wget https://zenodo.org/record/7596724/files/orca_leukemia_score.tar.gz
tar -xzvf  orca_leukemia_score.tar.gz -C ../resources
cd ../resources
mv orca_leukemia_score/* .
rm -r orca_leukemia_score
cd ../
 ```
 7. Download the Orca Leukemia repository
 ```
git clone https://github.com/jzhoulab/orca_leukemia.git
mv orca_leukemia/* .
rm -r orca_leukemia
```

## Usage
The script outputs a prediction file (prediction.pth), 3D regulatory impact scores for every TSS associated with this gene (output.csv) and 3D regulatory impact scores for the TSS with the highest gain of 3D regulatory impact score (output.short.csv).
<br/>
<br/>
<br/>
To run a scoring for the **translocation**:
```
./score.sh variant chrm1 breakpoint1 chrm2 breakpoint2 strands cell_type gene_name output_file
```
Example:
```
./score.sh TL chr22 42404617 chr12 57046743 -+ T-ALL_GSE134761 MYL6B TL_scores
```
<br/>

To get a scoring for the **inversion**:
```
./score.sh variant chrm1 breakpoint1 cell_type gene_name output_file
```
Example:
```
./score.sh INV chr8 107947105 107970736 ETP_GSE146901 ABRA INV_scores
```
<br/>

To get a scoring for the **deletion**:
```
./score.sh variant chrm1 breakpoint1 cell_type gene_name output_file
```
Example:
```
./score.sh DEL chr19 10894765 12246127  T-ALL_GSE134761 ICAM5 DEL_variant
```
<br/>

To get a scoring for the **duplication**:
```
./score.sh variant chrm1 breakpoint1 cell_type gene_name output_file
```
Example:
```
./score.sh DUP chr14 104681516 104682030 T-ALL_GSE134761 INF2 DUP_variant
```



