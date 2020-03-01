export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib:/usr/local/cuda-9.0/lib64
alias python="/usr/bin/python3.6"

cd ./code
python train.py