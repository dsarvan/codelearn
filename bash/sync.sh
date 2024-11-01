#!/usr/bin/env bash
# File: sync.sh
# Name: D.Saravanan
# Date: 11/02/2024
# Script to sync files using rsync

dpkg --get-selections > .gusdt/home/user/packages.txt

#rsync -av .bashrc .dotfiles/.bashrc
#rsync -av .bashrc .gusdt/home/user/.bashrc

rsync -av .config/user-dirs.dirs .dotfiles/.config/
rsync -av .config/user-dirs.dirs .gusdt/home/user/.config/

rsync -av /etc/doas.conf .dotfiles/etc/
rsync -av /etc/ly/config.ini .dotfiles/etc/ly/
rsync -av /etc/apt/sources.list .gusdt/etc/apt/
rsync -av /etc/asound.conf .gusdt/etc/
rsync -av /etc/default/grub .gusdt/etc/default/
rsync -av /etc/doas.conf .gusdt/etc/
rsync -av /etc/ly .gusdt/etc/
rsync -av /etc/network/interfaces .gusdt/etc/network/

# epy
rsync -av .config/epy/configuration.json .dotfiles/.config/epy/
rsync -av .config/epy/configuration.json .gusdt/home/user/.config/epy/

# qutebrowser
rsync -av .config/qutebrowser/config.py .dotfiles/.config/qutebrowser/
rsync -av .config/qutebrowser/config.py .gusdt/home/user/.config/qutebrowser/
rsync -av .config/qutebrowser/bookmarks .gusdt/home/user/.config/qutebrowser/

# vifm
rsync -av .config/vifm .dotfiles/.config/
rsync -av .config/vifm .gusdt/home/user/.config/

# yt-dlp
rsync -av .config/yt-dlp/config .dotfiles/.config/yt-dlp/
rsync -av .config/yt-dlp/config .gusdt/home/user/.config/yt-dlp/

# zathura
rsync -av .config/zathura/zathurarc .dotfiles/.config/zathura/
rsync -av .config/zathura/zathurarc .gusdt/home/user/.config/zathura/

# dmenu
rsync -av .dmenu/config.h .dotfiles/.dmenu/
rsync -av --exclude '.git' .dmenu .gusdt/home/user/

# dwm
rsync -av .dwm/config.h .dotfiles/.dwm/
rsync -av --exclude '.git' .dwm .gusdt/home/user/

# gitconfig
rsync -av .gitconfig .dotfiles/
rsync -av .gitconfig .gusdt/home/user/

# gnuplot
rsync -av .gnuplot .dotfiles/
rsync -av .gnuplot .gusdt/home/user/

# sent
rsync -av .sent/config.h .dotfiles/.sent/
rsync -av --exclude '.git' --exclude '.gitignore' .sent .gusdt/home/user/

# slstatus
rsync -av .slstatus/config.h .dotfiles/.slstatus/
rsync -av --exclude '.git' .slstatus .gusdt/home/user/

# st
rsync -av .st/config.h .dotfiles/.st/
rsync -av .st/st-expected-anysize-0.9.diff .dotfiles/.st/
rsync -av --exclude '.git' .st .gusdt/home/user/

# tmux
rsync -av .tmux.conf .dotfiles/
rsync -av .tmux.conf .gusdt/home/user/

# vim
rsync -av .vim/templates .dotfiles/.vim/
rsync -av .vim/templates .gusdt/home/user/.vim/

# vimrc
rsync -av .vimrc .dotfiles/.vimrc
rsync -av .vimrc .gusdt/home/user/.vimrc

# w3m
rsync -av .w3m/config .dotfiles/.w3m/config
rsync -av .w3m/keymap .dotfiles/.w3m/keymap
rsync -av .w3m/config .gusdt/home/user/.w3m/config
rsync -av .w3m/keymap .gusdt/home/user/.w3m/keymap
rsync -av .w3m/bookmark.html .gusdt/home/user/.w3m/bookmark.html
