# Ubuntu 18.04

adduser --disabled-password --gecos "" cns

usermod -aG sudo cns
echo "cns  ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

sudo su -- cns
mkdir ~/.ssh/
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDUC/ybly1FFaKgQguBspveCylCrlNxM3UdD1jVcTarblcNbJ8cwiOB1qzKtO7SdZDpAfR2egYwzhAO60sWFRrIRy6cKs0V1EcUIDyKUt8wRXeIx7FG94gFo6AnlE69m6Nx47Qf6ZF+3p97ebKwup5Az6Ckg2qyb2lli7XFza119ONyB9enczMahCb+DCgkrotXz0Be5Ws+fe34sm2GgSmv62QeBYcykfUqCzVkoHAGR1zY2TegtyKIzkgxky47pvYeX0zfB/WgJWPJrrspCZ98H8DgeJh++3YMHUlV60Zu39f9Y7CTUN58bmStDnzK+inepYpLn6/+Iiy2iBm0kxqP" > ~/.ssh/authorized_keys

echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -c -s)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update

sudo add-apt-repository -y ppa:timescale/timescaledb-ppa
sudo apt-get update
sudo apt install -y timescaledb-postgresql-11
sudo timescaledb-tune -yes

exit

echo "listen_addresses = '*'" >> /etc/postgresql/11/main/postgresql.conf
echo "host    all             cns_user        0.0.0.0/0                       md5" >> /etc/postgresql/11/main/pg_hba.conf
echo "host    all             cns_user        ::/0                            md5" >> /etc/postgresql/11/main/pg_hba.conf
sed -i  '/^local   all             all                                     peer/ s/peer/md5/' /etc/postgresql/11/main/pg_hba.conf
sed -i  '/^local   all             postgres                                peer/ s/peer/md5/' /etc/postgresql/11/main/pg_hba.conf
service postgresql restart
