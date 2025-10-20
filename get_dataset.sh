# Download dataset
gdown https://drive.google.com/drive/folders/1XHJRi65ISZi4OPF_SbxQaxVsXFc7D24v?usp=drive_link -O data --folder

# unzip the dataset and rename to data
unzip -o -O UTF-8 "data/40_初賽資料_V3 1.zip" -d data
mv data/初賽資料/*.csv data/
rm -rf data/初賽資料
rm "data/40_初賽資料_V3 1.zip"
