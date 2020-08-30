#!/usr/bin/env bash

# update and install sudo
apt-get update
apt-get install -y sudo

# add user
##user="raman"
##useradd -m $user
# set password
##password=uiop[]
##echo -e "$password\n$passowrd\n" | sudo passwd $user

# add sudo access
##usermod -aG sudo $user

# replace sh shell to bash shell
##var=`grep "^$user" /etc/passwd | cut -d : -f 7`

##if [ $var = "/bin/sh" ]
##then
##	sed -i -e '/raman/s/bin\/sh/bin\/bash/g' /etc/passwd
##fi

user="raman"
password="uiop[]"

useradd -m -p $password -s /bin/bash -G sudo $user

# switch user and execute

sudo -i -u $user bash << EOF

# ssh configuration
sudo apt-get update
sudo apt-get install -y vim openssh-server ssh
mkdir /home/$user/.ssh/
sudo ssh-keygen -t rsa -P " " -f "/home/$user/.ssh/id_rsa" -q
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# vi configuration
if [ -f /home/$user/.exrc ]
then
	echo "set autowrite
	set hidden
	set ignorecase
	set incsearch
	set nomesg
	set number
	set shiftwidth=4
	set showcmd
	set showmatch
	set smartcase
	set tabstop=4
	highlight LineNr cterm=NONE ctermfg=DarkGrey ctermbg=NONE" >> /home/$user/.exrc
else
	touch /home/$user/.exrc
	echo "set autowrite
	set hidden
	set ignorecase
	set incsearch
	set nomesg
	set number
	set shiftwidth=4
	set showcmd
	set showmatch
	set smartcase
	set tabstop=4
	highlight LineNr cterm=NONE ctermfg=DarkGrey ctermbg=NONE" >> /home/$user/.exrc
fi

# start ssh
# sysVinit
sudo service ssh start
# systemd
#sudo systemctl start ssh

# installing openjdk8
# updating the packages list
sudo apt-get update
# installing the dependencies necessary to add a new repository over HTTPS
sudo apt-get install -y apt-transport-https ca-certificates wget dirmngr gnupg software-properties-common
# import the repository's GPG key
wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | sudo apt-key add -
# add the adoptopenjdk apt repository to the system
sudo add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
# update apt sources and install Java 8
sudo apt-get update && apt-get install -y adoptopenjdk-8-hotspot
# JAVA_HOME environment variable
sudo echo "JAVA_HOME=/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64" >> /etc/environment
# for changes to take effect on your current shell
source /etc/environment

# hadoop configuration
wget https://archive.apache.org/dist/hadoop/common/hadoop-3.2.1/hadoop-3.2.1.tar.gz
tar -xzf hadoop-3.2.1.tar.gz
mv hadoop-3.2.1/ /usr/local/hadoop/ 

# set environment variables
echo "PATH=/usr/local/hadoop/bin:/usr/local/hadoop/sbin:$PATH" >> /home/$user/.profile
echo "export JAVA_HOME=/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_HDFS_HOME=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin" >> /home/$user/.bashrc
source .bashrc

# configure the node
# number of replications
replica=2

# hdfs-site.xml
sed -i "s/<configuration>/& \
\n\t<property> \
\t\t<name>dfs.namenode.name.dir<\/name> \
\t\t<value>/usr/local/hadoop/hdfs/nameNode<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>dfs.datanode.data.dir<\/name> \
\t\t<value>/usr/local/hadoop/hdfs/dataNode<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>dfs.replication<\/name> \
\t\t<value>$replica<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>dfs.permission.enabled<\/name> \
\t\t<value>true<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>dfs.namenode.acls.enabled<\/name> \
\t\t<value>true<\/value> \
\t<\/property>/" /usr/local/hadoop/etc/hadoop/hdfs-site.xml

# core-site.xml
sed -i "s/<configuration>/& \
\n\t<property> \
\t\t<name>dfs.default.name<\/name> \
\t\t<value>hdfs://node-master:9000<\/value> \
\t<\/property>/" /usr/local/hadoop/etc/hadoop/core-site.xml

# mapred-site.xml
sed -i "s/<configuration>/& \
\n\t<property> \
\t\t<name>mapreduce.framework.name<\/name> \
\t\t<value>yarn<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>yarn.app.mapreduce.am.env<\/name> \
\t\t<value>HADOOP_MAPRED_HOME=$HADOOP_HOME<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>mapreduce.map.env<\/name> \
\t\t<value>HADOOP_MAPRED_HOME=$HADOOP_HOME<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>mapreduce.reduce.env<\/name> \
\t\t<value>HADOOP_MAPRED_HOME=$HADOOP_HOME<\/value> \
\t<\/property>/" /usr/local/hadoop/etc/hadoop/mapred-site.xml
EOF
