mkdir ~/jq
cd ~/jq
wget https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
chmod +x jq-linux64
echo "alias jq="$(pwd)"/jq-linux64" >> ~/.bashrc
source ~/.bashrc
cd -

