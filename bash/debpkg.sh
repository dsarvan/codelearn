#!/usr/bin/env bash
# File: debpkg.sh
# Name: D.Saravanan
# Script to install debian packages and suckless programs

sucks='dwm st slstatus dmenu'

for suck in $sucks; do
	git clone https://git.suckless.org/"$suck"
done

packages='x11-xserver-utils git make gcc libx11-dev libxft-dev libxinerama-dev
xorg unclutter scrot sxiv build-essential texlive-full zathura zathura-ps
zathura-djvu vim ffmpeg mpv alsa-utils w3m pandoc qutebrowser indent wordnet'

for package in $packages; do
	doas apt-get install -y "$package"
done

doas apt-get remove -y wordnet-gui
