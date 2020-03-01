alias python='/home/yuanmingkuan/anaconda3-bak/bin/python3'

cd ./code
python resize_images.py
python dataset_tool.py create_cub ../data/cub/train/tfrecord ../data/cub