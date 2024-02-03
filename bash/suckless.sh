#!/usr/bin/env bash
# File: suckless.sh
# Name: D.Saravanan
# Date: 02/01/2021
# Script to install suckless programs

export OS=$(uname -s)

suckless() {

	program=$1

	if [ -d $program ]; then
		cd $program
		git pull
	else
		git clone https://git.suckless.org/$program
		cd $program
	fi

	cd
}

for program in dmenu dwm slock slstatus st surf; do
	suckless $program
done
