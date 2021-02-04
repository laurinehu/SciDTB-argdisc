# SciDTB-argdisc

This repository contains the data and code for the paper : 

```
Assessing Argumentation Structure Prediction from Discourse Structure: a Symbolic Approach.
```

## Overview

- Data: SciDTB corpus and argumentation annotations provided by Accuosto et al. 
- Analysis: Analysis of the correspondances between Discourse and Argumentation Structures
- Prediction: Prediction and evaluation of the obtained predictions
- Utils: contains modules used in analysis and prediction notebooks

## Directories

### Data

- dev/train/test correspond to SciDTB annotated corpus, as organized by the authors for there experiments
- scidtb_argmin_annotations are argumentation annotations of the 60 documents provided by Accuosto

### Analysis

- jupyter notebook organized as in the paper

### Prediction

- jupyter notebook used to predict and evaluate the prediction (scores reported in the paper)

### Utils

- all modules used for mining arg and disc graphs
- gspan module (fork from https://github.com/betterenvi/gSpan that we slighlty modified to deal with nxgraphs)
