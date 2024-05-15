#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

if [ ! -e "$1" ]; then
	echo "please enter the input file"
	exit 1	
fi

echo "------------------" >> $2
echo "Unique cities in the given data file:" >> $2
awk -F, 'NR>1 {print $3} ' $1 | sort -u >> $2
echo "------------------" >> $2

echo "Details of top 3 individuals with the highest salary:" >> $2
sort -t, -k4 -r $1 > temp.csv
awk 'NR>1 && NR<5 { print $0 }' temp.csv | sort -t, -k1 >> $2
echo "------------------" >> $2


echo "Details of average salary of each city:" >> $2
awk -F, 'NR>1 {sum[$3]+=$4; count[$3]++} END {for (city in sum) {avg = sum[city]/count[city]; printf "City:%s, Salary: %s\n",city, (avg == int(avg) ? sprintf("%.0f", avg) : sprintf("%.1f", avg))}}' "$1" >> "$2"


echo "------------------" >> $2
echo "Details of individuals with a salary above the overall average salary:" >> $2
average=$(awk -F, 'NR>1 {sum+=$4; count++} END{print sum/count}' $1)

awk -F, -v average="$average" 'NR>1 {if($4 > average) {gsub(/,/,"",$0); print $0}}' "$1" >> "$2"

