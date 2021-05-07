#!/bin/bash
./no_vec $1 $1 $1 $2
./auto_vec $1 $1 $1 $2
./gather $1 $1 $1 $2
./alignr $1 $1 $1 $2
./switch $1 $1 $1 $2
./alignr_switch $1 $1 $1 $2
