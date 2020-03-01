cd ./evaluation

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib:/usr/local/cuda-9.0/lib64
alias python='/usr/bin/python3.6'
model_path='../code/results/00000-bgan-cub-cond-2gpu/network-snapshot-010526.pkl'
CUDA_VISIBLE_DEVICES=0 python generate_images.py $model_path

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib:/usr/local/cuda-8.0/lib64
alias python='/usr/bin/python2.7'
images_path='./generated_images.npy'
CUDA_VISIBLE_DEVICES=0 python inception_score.py $images_path