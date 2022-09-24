---
layout: single
title: "Configuration for VSCode and ZSH in Remote Server"



---



### Installation when beginning

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
apt install g++ -y
apt-get install build-essentials
apt-get install curl
```

### Additional customization for bash

1. Install sudo
   https://unix.stackexchange.com/questions/354928/bash-sudo-command-not-found

   ```bash
   su -
   apt-get install sudo -y
   usermod -aG sudo root
   ```

2. Install zsh first
   https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH#install-and-set-up-zsh-as-default

   ```bash
   sudo apt install zsh
   ```

3. Make zsh default shell, and also on ssh VSCode

   - https://askubuntu.com/questions/131823/how-to-make-zsh-the-default-shell

   ```bash
   chsh -s $(which zsh)
   ```

   - https://code.visualstudio.com/docs/editor/integrated-terminal#_terminal-profiles

   ```json
   // terminal settings
    "terminal.integrated.defaultProfile.linux": "zsh",
   ```

4. Install oh-my-zsh
   
   ```bash
   sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
   
   ```
   
5. Oh-my-zsh configuration: https://l-yohai.github.io/VSCode-Terminal-Customizing/#more

6. make some space for zsh configuration wizard

   ```zsh
   typeset -i10 COLUMNS=50
   typeset -i10 LINES=23
   ```

7. run configuration wizard with the following instruction
   https://github.com/romkatv/powerlevel10k#configuration-wizard

### how to connect github remote with upstage server

After initializing zsh, you have to register github 2fa for ssh

```bash
git clone https://github.com/boostcampaitech2/klue-level2-nlp-15
```

```bash
git config --global user.email "<useremail>"
git config --global user.name "<username>"
```

### empty trash bins

```bash
rm -rf ~/.local/share/Trash/*
```