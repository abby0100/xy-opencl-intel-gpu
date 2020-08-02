#!/bin/bash

target=xydistance
deviceid=$1
debug=$2

	if [[ "0" == "$debug" ]] || [[ -z "$debug" ]]; then
		cmake .. -DDEBUG=OFF
	else
		cmake .. -DDEBUG=ON
	fi

	make -j

	# run
	export OMP_NUM_THREADS=8
	numactl -C 0-7 ./$target $deviceid

# usage:
# ./compile-and-run.sh 0 | grep Kernel
# ./compile-and-run.sh 1 | grep Kernel
