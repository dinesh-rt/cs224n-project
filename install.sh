sudo dnf install nginx curl supervisor -y
sudo service supervisor start


jupyter notebook --generate-config

ipAdd=`curl http://169.254.169.254/latest/meta-data/public-ipv4`
echo "${ipAdd}"
