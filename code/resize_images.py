import os, shutil, pickle, pdb, cv2

with open('../data/cub/train/filenames.pickle', 'rb') as f:
	train_file = pickle.load(f)

if not os.path.exists('../data/cub/train/images_256/'):
	os.mkdir('../data/cub/train/images_256/')

for item in train_file:	
	filename = item.split('/')[-1] + '.jpg'
	oripath = os.path.join('../data/cub/images/', item + '.jpg')
	newpath = os.path.join('../data/cub/train/images_256/', filename)
	
	img = cv2.imread(oripath)
	resized_img = cv2.resize(img, (256, 256))
	
	cv2.imwrite(newpath, resized_img)