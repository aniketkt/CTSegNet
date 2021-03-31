#!/bin/bash

data_path=$1
#'../data'
fnames=$1'/*'$2'*'

for fpath in $fnames
do
#	echo $fpath
	fname=${fpath##*/}
	basename=${fname%$2}
#	echo $fname
#	echo $basename
#	echo $data_path
	
	if [ -d $fpath ]; then
		echo "Working on: "$fname
		python bin/rw_utils/convert_to_hdf5.py -f $fpath -o $data_path -n $basename -c 1.0 -x data -y -d
	fi
done

#python code/bin/rw_utils/convert_to_hdf5.py -f data/5L_load00_rec_1x1_16bit_tiff -i
