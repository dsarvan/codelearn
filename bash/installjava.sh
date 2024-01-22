#!/usr/bin/env bash
# bash script to install Java LTS version 8 in Debian 10 Buster

# updating the packages list
sudo apt-get update

# installing the dependencies necessary to add a new repository over HTTPS
sudo apt-get install apt-transport-https ca-certificates wget
sudo apt-get install dirmngr gnupg software-properties-common

# import the repository's GPG key using the following wget commands
wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | sudo apt-key add -

# add the AdoptOpenJDK APT repository to the system
sudo add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/

# once the repository is enabled, update apt sources and install Java 8
sudo apt-get update
sudo apt-get install adoptopenjdk-8-hotspot

# verify the installation by checking the Java version
java -version
