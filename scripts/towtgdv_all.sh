##############################################################################################
# Purpose: Network to wt gdv
# Author: Khalique Newaz
#
# The parameter [input-dir] is the directory containg the folder that has all the networks
#
# The parameter [annotation-file] is the file containg the list of proteins whose GDVs are calculated
#
# The parameter [algorithm] has four values: 1, 2, 3, and 4, that correspond to the following.
# 1: edge vs. edge orbits COUNTS
# 2: edge vs. edge orbits WEIGHTS
# 3: node vs. edge orbits COUNTS
# 4: node vs. edge orbits WEIGHTS
#
# 
##############################################################################################

#!/usr/bin/bash
if [ "$#" -lt 4 ]; then
     echo Error: Wrong number of arguments
     echo Usage: $0 [input-dir] [annotation-file] [algorithm] [output-dir] -m [atom: default 4]
     echo Example: $0 [input-dir] [annotation-file] 2 [output-dir] -m 4
     exit
     fi
     
     totalArgument=$#
       
     matDirectory=$1
     shift
     annotationfile=$1
     shift
     algorithm=$1
     shift
     outputDir=$1
     #outputDir="${outputDir,,}"
     shift
     
     if [ ! -d $matDirectory ]; then
     echo Error: Directory $matDirectory not found
     exit
     fi
     
     
     b=$(basename $annotationfile | cut -d"." -f1)
     
     ## default values ###############################
     
     cutoff=4
     
     ##################################################

     for((i=5;i<=totalArgument;i+=2)); do
     type=$1
     shift
     if [ $i -eq $totalArgument ]; then
     echo Error: Missing value for the parameter $type
     exit
     fi
     val=$1
     shift
     
     if [ "$type" == "-m" ]; then
     cutoff=$val
     else
       echo Error: Wrong type of parameter $type
     exit
     fi
     done
	 
	 if [ ! -d $outputDir ]; then
     mkdir $outputDir
     fi
     
     if [ ! -d $outputDir/wt-GDVs-$cutoff-A ]; then
     mkdir $outputDir/wt-GDVs-$cutoff-A
     fi
          
     outt=$matDirectory/
     
     IFS=$'\n'
     
     ###########################################################################
     # computing wt edge gdvs
     
     # for every id in the annotation file
     count=0
     for line in `cat $annotationfile`;
     do
        filID=$(echo $line | cut -f2)
        fileID=$(echo $filID | cut -d'.' -f 1)
        #echo $fileID
     
        goutt=$outputDir/wt-GDVs-$cutoff-A/$fileID
     
        if [ ! -d  $goutt ]; then
          mkdir $goutt
        fi
     
        if [ $algorithm == 1 ];
        then
            ./scripts/bin/ecount $outt/$fileID.gw $goutt/ew
        elif [ $algorithm == 2 ];
        then 
            ./scripts/bin/ewcount $outt/$fileID.gw $goutt/ew
        elif [ $algorithm == 3 ];
        then
            ./scripts/bin/necount $outt/$fileID.gw $goutt/ew
        elif [ $algorithm == 4 ];
        then
            ./scripts/bin/nwcount $outt/$fileID.gw $goutt/ew
        fi
     
        count=$((count + 1))
        echo  computing weighted GDVs... $count
     done
     
     
     
     