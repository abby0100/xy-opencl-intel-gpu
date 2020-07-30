#!/bin/bash

target=xydistance
deviceid=$1

	cmake ..
	make -j
	./$target $deviceid

# usage:
# ./compile-and-run.sh 0 | grep Kernel
# ./compile-and-run.sh 1 | grep Kernel
