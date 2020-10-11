

# Download submissions
mkdir wmt
cd wmt

# WMT19
wget http://data.statmt.org/wmt19/translation-task/wmt19-submitted-data-v3.tgz .
wget https://www.computing.dcu.ie/~ygraham/newstest2019-humaneval.tar.gz .

tar -xzvf wmt19-submitted-data-v3.tgz
tar -xzvf newstest2019-humaneval.tar.gz

# WMT18
wget http://data.statmt.org/wmt18/translation-task/wmt18-submitted-data-v1.0.1.tgz .
wget http://computing.dcu.ie/~ygraham/newstest2018-humaneval.tar.gz .

tar -xzvf wmt18-submitted-data-v1.0.1.tgz
tar -xzvf newstest2018-humaneval.tar.gz


# Cleanup
rm *.tgz
rm *.tar.gz
