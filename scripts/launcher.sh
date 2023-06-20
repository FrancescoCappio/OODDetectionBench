#!/bin/bash 

list_file=$1
select_line=$2

line=$(awk "NR==$select_line" $list_file)

eval $line 
