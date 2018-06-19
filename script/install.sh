#!/usr/bin/env bash

# git clone https://github.com/dimitrijejankov/protoDB.git

sudo apt-get update
sudo apt-get install -y cmake mpich libgsl-dev libboost-all-dev libatomic-ops-dev libatomic1
sudo add-apt-repository ppa:jonathonf/gcc-7.1
sudo apt-get update
sudo apt-get install -y gcc-7 g++-7