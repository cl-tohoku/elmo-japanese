#!/bin/sh

cd data
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar zxvf ldcc-20140209.tar.gz
python extract_sents.py --dir_path text
mecab -O wakati dataset.sent.txt -o dataset.wakati.txt -b 50000
paste dataset.wakati.txt dataset.label.txt > dataset.wakati-label.txt