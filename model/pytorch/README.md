# PyTorch setup

For CPU training:
```
mkvirtualenv miniplaces
pip install -r requirements.txt
pip install ipython jupyter  # optional
pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl  # depends on your system. See http://pytorch.org/.
pip install torchvision
```

On AWS CPU:
```
anaconda3/bin/conda install pytorch torchvision -c soumith
```

On AWS GPU:
```
conda install pytorch torchvision cuda80 -c soumith
```

Repository setup:
```
ssh-keygen -t rsa -b 4096 -C "youremail@mit.edu"  # and add the key to your Github account

git clone git@github.com:ajayjain/miniplaces.git

cd miniplaces/data
wget http://6.869.csail.mit.edu/fa17/miniplaces/data.tar.gz
tar -xvf data.tar.gz
```








```
python3 -m venv ~/
```

```
set -e

# Set up pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
echo 'export PATH="/home/ubuntu/.pyenv/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
source ~/.bash_profile
pyenv update
```