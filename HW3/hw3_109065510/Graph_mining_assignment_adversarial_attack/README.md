# 請先安裝好anaconda & cuda 11.1

# Usage
```
git clone https://gitlab.com/warren30815/Graph_mining_assignment_adversarial_attack.git
cd Graph_mining_assignment_adversarial_attack
conda create --name nettack python=3.7 -y
conda activate nettack
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip3 install -r requirements.txt
python3 main.py
```
