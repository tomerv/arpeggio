#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: runflow.sh <dataset_name>"
    exit 1
fi


#######
# Generate dataset
#######

DATASET_NAME=$1
echo "DATASET: $DATASET_NAME"
# python3 ${DATASET_NAME}.py

#######
# Run models
#######

if [ -e results-$DATASET_NAME ]; then
    results_backup=results-${DATASET_NAME}-OLD$(date +'%s')
    echo "Moving existing results directory to $results_backup"
    mv results-$DATASET_NAME $results_backup
fi
mkdir results-$DATASET_NAME


#######
# Baseline: popularity
#######

MODEL_NAME='popularity'
date
echo "MODEL: $MODEL_NAME"
python3 baseline-popularity.py $DATASET_NAME
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date


#######
# Baseline: BPR
#######

MODEL_NAME='bpr'
date
echo "MODEL: $MODEL_NAME"
python3 baseline-bpr.py $DATASET_NAME
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date

#######
# Ours
#######

MODEL_NAME='ours'
date
echo "MODEL: $MODEL_NAME"
python3 ours.py $DATASET_NAME
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date


#######
# Ours2
#######

MODEL_NAME='ours2'
date
echo "MODEL: $MODEL_NAME"
python3 ours2.py $DATASET_NAME
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date

exit 0 #TODO

function train_matlab_model {
    DATASET_NAME=$1
    MODEL_NAME=$2
    MODEL_DIR=$3
    MATLAB_COMMAND=$4
    matlab -nodisplay -r \
"Traindata=readmat('${DATASET_NAME}.train.txt');"\
"Testdata=readmat('${DATASET_NAME}.val.txt');"\
"olddir=cd('$MODEL_DIR');"\
"$MATLAB_COMMAND;"\
"cd(olddir);"\
"dlmwrite('results-$DATASET_NAME/${MODEL_NAME}.u.txt', U);"\
"dlmwrite('results-$DATASET_NAME/${MODEL_NAME}.v.txt', V);"\
"quit"
}


#######
# CLiMF
#######

MODEL_NAME='climf'
date
echo "MODEL: $MODEL_NAME"
train_matlab_model ${DATASET_NAME} ${MODEL_NAME} '/home/tvromen/research/CLiMF_code' 'climf'
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date


#######
# UOCCF (CLiMF + PMF)
#######

MODEL_NAME='uoccf'
date
echo "MODEL: $MODEL_NAME"
train_matlab_model ${DATASET_NAME} ${MODEL_NAME} '/home/tvromen/research/CLiMF_code' 'uoccf'
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date

#######
# ListRank
#######

MODEL_NAME='listrank'
date
echo "MODEL: $MODEL_NAME"
train_matlab_model ${DATASET_NAME} ${MODEL_NAME} '/home/tvromen/research/ListRank' '[U,V]=listrank(Traindata, 5, 0.01, 250, 1e-6)'
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date


#######
# URM (ListRank + PMF)
#######

MODEL_NAME='urm'
date
echo "MODEL: $MODEL_NAME"
train_matlab_model ${DATASET_NAME} ${MODEL_NAME} '/home/tvromen/research/ListRank' '[U,V]=URM(Traindata, 5, 0.01, 250, 1e-6)'
python3 calc_scores.py ${DATASET_NAME} $MODEL_NAME
date


