import os
def mkdir(p):
  if not os.path.exists(p):
    os.mkdir(p)

def link(src,dst):
  if not os.path.exists(dst):
    os.symlink(src,dst,target_is_directory=True)
    
mkdir('C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360_dataset/fruits-360')
classes = [
  'Apple Golden 1',
  'Avocado',
  'Lemon',
  'Mango',
  'Kiwi',
  'Banana',
  'Strawberry',
  'Raspberry'
]

train_path_from = os.path.abspath('C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360_dataset/fruits-360/training')
test_path_from = os.path.abspath('C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360_dataset/fruits-360/test')

train_path_to = os.path.abspath('C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360-original-size/fruits-360-original-size\Training')
test_path_to = os.path.abspath('C:/Users/Emmanuel Alo/Documents/vgg data/fruits-360-original-size/fruits-360-original-size/Validation')

mkdir(train_path_to)
mkdir(test_path_to)