#!/bin/bash
/mnt/c/Users/rings/Miniconda3/python.exe dCount.py |tee log.dat 
    
    #sed -r -n "s/.*Training loss\: ([0-9]*\.[0-9]*).*Validation loss\: ([0-9]\.[0-9]*).*/\1 \2/p" 
