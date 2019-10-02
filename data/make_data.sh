#!/bin/sh

wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar zxvf ldcc-20140209.tar.gz


mecab -O wakati dataset.sent.txt -o dataset.wakati.txt -b 50000
paste dataset.wakati.txt dataset.label.txt > dataset.wakati-label.txt