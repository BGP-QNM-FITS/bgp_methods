#!/bin/bash

filepath='/data/rvnd2-2/CCE_data/raw_data' 

cd
cd $filepath

declare -a IDs=(
10783245
#10783515
#10783526
#10783534
#10783547
#10783566
#10783575
#10783582
#10783596
#10783609
#10783616
#10783631
#10783640
)

for id in "${IDs[@]}"
do
   mkdir $id
   cd $id    
   mkdir Lev2
   mkdir Lev3
   mkdir Lev4
   mkdir Lev5
   zenodo_get $id 
   cd .. 
done