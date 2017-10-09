# PyTorch setup

For CPU training:
```
mkvirtualenv miniplaces
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl  # depends on your system. See http://pytorch.org/.
pip install torchvision
```

On AWS CPU:
```
~/src/anaconda3/bin/conda install pytorch torchvision -c soumith
```

On AWS GPU:
```
~/src/anaconda3/bin/conda install pytorch torchvision cuda80 -c soumith
```

Add this to your ~/.bash_profile:
```
# Anaconda 3 setup
export PATH="/home/ubuntu/src/anaconda3/bin:$PATH"
export PYTHONPATH="/home/ubuntu/src/anaconda3/pkgs"
```

Repository setup:
```
ssh-keygen -t rsa -b 4096 -C "youremail@mit.edu"  # and add the key to your Github account

git clone git@github.com:ajayjain/miniplaces.git

cd miniplaces/data
wget http://6.869.csail.mit.edu/fa17/miniplaces/data.tar.gz
tar -xvf data.tar.gz

cd ../model/pytorch
python prepro_data.py
```
