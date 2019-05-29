#!/bin/bash
# Basic if statement

VAL=$1
DATA=${PWD}'/data'

if [ $1 = 1 ] 
then
	echo Executing test and training $1 
	python prediction.py 
fi

if [ $1 = 2 ]
then
	echo Executing test and training $1
	python prediction.py ${DATA}/training.txt ${DATA}/test${VAL}.txt
fi

if [ $1 = 3 ]
then
	echo Executing test and training $1
	python prediction.py ${DATA}/training.txt ${DATA}/test${VAL}.txt
fi

