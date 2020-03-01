alias python='/usr/bin/python3.6'

cd ./code
python resize_images.py
python dataset_tool.py create_cub ../data/cub/train/tfrecord ../data/cub