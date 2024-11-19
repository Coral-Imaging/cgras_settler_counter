#!/bin/bash

#PBS -N long_train
#PBS -l walltime=01:00:00
#PBS -l ncpus=3
#PBS -l mem=16gb
#PBS -l ngpus=1
#PBS -l gputype=A100
#PBS -m abe
#PBS -I