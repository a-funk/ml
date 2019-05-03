#!/bin/bash
# Basic if statement

VAL=$1
DATA='/Users/alexfunk/code/ml/hw2_starter_package/data'

if [ $1 = 1 ] 
then
	echo Executing test and training $1 
	python hw2.py ${DATA}/training${VAL}.txt ${DATA}/testing${VAL}.txt
fi

if [ $1 = 2 ]
then
	echo Executing test and training $1
	python hw2.py ${DATA}/training${VAL}.txt ${DATA}/testing${VAL}.txt
fi

if [ $1 = 3 ]
then
	echo Executing test and training $1
	python hw2.py ${DATA}/training${VAL}.txt ${DATA}/testing${VAL}.txt
fi

