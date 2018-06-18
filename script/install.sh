#!/usr/bin/env bash

# git clone https://github.com/dimitrijejankov/protoDB.git

sudo apt-get update
sudo apt-get install -y mpich
sudo apt-get install -y libgsl
sudo apt-get install -y libgsl-dev
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y libatomic-ops-dev
sudo apt-get install -y libatomic-dev
sudo add-apt-repository ppa:jonathonf/gcc-7.1
sudo apt-get update
sudo apt-get install -y gcc-7 g++-7