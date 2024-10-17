#!/bin/bash
wget https://figshare.com/ndownloader/files/34683070
unzip 34683070 -d p19_unzipped && rm 34683070
mkdir -p data/p19
cp -r p19_unzipped/P19data/processed_data data/p19
cp -r p19_unzipped/P19data/splits data/p19
rm -rf p19_unzipped