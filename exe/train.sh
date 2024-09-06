# clone model sentence-transformers, tiến hành truy xuất
huggingface-cli login --token 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
cd ..
git clone https://github.com/UKPLab/sentence-transformers.git
cd sentence-transformers
pip install -e .
cd ../Fake-Real-News-Classification
pip install -r requirement.txt
mkdir data
cd data
gdown 1zCjH1YsGgEVgegTl-RVOMwaEy_7p3Uht
python -m zipfile -e Dataset-Fake-Real-News-Gathering.zip .
python ./src/data_transform.py
python ./src/train-sentence-BERT.py
