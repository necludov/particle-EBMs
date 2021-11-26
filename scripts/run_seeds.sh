#!/bin/bash
for s in `seq 0 9`
do
python $1 --seed $s
done