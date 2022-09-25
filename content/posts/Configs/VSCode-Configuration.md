---
title: "My VSCode editor settings in JSON format"
date: 2021-09-20
tags: ["Configs"]
draft: false
---

I had to reconfigure VSCode editor while connecting to the GPU server using ssh.

The following configuration has 3 objectives.

- I had to use both Prettier formatter and Black formatter. Prettier formatter is convinient for JSON and other files whereas Black is the popular Python formatter used among Korean developers. So I had to use both of those formatters without collision.
- Reduce the overload on the VSCode for the purpose of saving memory.
- Remove unnecessary folders and files from the sidebar.

```json
{
  "terminal.integrated.defaultProfile.linux": "zsh",
  "editor.fontSize": 16,
  "editor.tabSize": 2,

  // python extension settings
  "python.defaultInterpreterPath": "/home/noopy/anaconda3/envs/gcpqa/bin/python3",

  "python.languageServer": "Pylance",

  // default formatting settings
  "editor.defaultFormatter": "esbenp.prettier-vscode", // default formatter is prettier
  "[python]": {
    "editor.defaultFormatter": "ms-python.python" // pytthon default formatter is black
  },
  "editor.formatOnSave": true,
  "editor.formatOnSaveMode": "file",

  // python formatter settings
  "python.formatting.provider": "black", // you should pip install black to the corresponding python environment first!
  "python.formatting.blackPath": "/home/noopy/anaconda3/bin/black",
  "python.formatting.blackArgs": ["--line-length", "100"],

  // exclude files on the search result: saving editor app's memory
  "files.watcherExclude": {
    // ignore default folders provided by upstage
    "**/.cache/**": true,
    "**/data/**": true,
    ".config": true,
    ".ipynb_checkpoints/**": true,
    ".ipython/**": true,
    // "**/.jupyter/**": true,
    ".local/**": true,
    ".nv/**": true,
    ".vscode-server/**": true,
    "**/input/**": true, // ignoring image dataset folder directory

    // ignore default files provided by upstage
    ".bash_history": true,
    // ".bashrc": true,
    ".profile": true,
    ".wget-hsts": true,

    // files to exclude
    "**/.DS_Store": true,
    "**/.git": true,
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/node_modules": true,

    // other folders to exclude
    "**/docker": true,
    "**/anaconda3": true,
    "**/.oh-my-zsh": true,
    "**/.gnupg": true,
    "**/.didim365": true
  },

  // hide files & directories from the sidebar for better look
  "files.exclude": {
    // ignore default folders provided by upstage
    "**/.cache/**": true,
    // "**/data/**": true,
    ".config": true,
    ".ipynb_checkpoints/**": true,
    ".ipython/**": true,
    // "**/.jupyter/**": true,
    ".local/**": true,
    ".nv/**": true,
    ".vscode-server/**": true,
    "**/input/**": true, // ignoring image dataset folder directory

    // ignore default files provided by upstage
    ".bash_history": true,
    // ".bashrc": true,
    ".profile": true,
    ".wget-hsts": true,
    ".python_history": true,
    // ".bash_profile": true,
    ".sudo_as_admin_successful": true,
    ".viminfo": true,
    ".Xauthority": true,
    ".zshrc.pre-oh-my-zsh": true,
    "**/.keras": true,

    // files to exclude
    "**/.DS_Store": true,
    "**/.git": true,
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/node_modules": true,
    "**.wakatime": true,
    "**.wakatime.**": true,
    "**.wakatime-internal.cfg": true,
    "**.zcompdump": true,
    "**.shell.pre-oh-my-zsh": true,
    // "**.zshrc": true,
    "**.zsh_history": true,
    "**.p10k.**": true,

    // other folders to exclude
    "**/docker": true,
    "**/anaconda3": true,
    "**/.oh-my-zsh": true,
    "**/.gnupg": true,
    "**/.didim365": true
  },

  "search.exclude": {
    // ignore default folders provided by upstage
    "**/.cache/**": true,
    "**/data/**": true,
    ".config": true,
    ".ipynb_checkpoints/**": true,
    ".ipython/**": true,
    // "**/.jupyter/**": true,
    ".local/**": true,
    ".nv/**": true,
    ".vscode-server/**": true,
    "**/input/**": true, // ignoring image dataset folder directory

    // ignore default files provided by upstage
    ".bash_history": true,
    // ".bashrc": true,
    ".profile": true,
    ".wget-hsts": true,

    // files to exclude
    "**/.DS_Store": true,
    "**/.git": true,
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/node_modules": true,

    // other folders to exclude
    "**/docker": true,
    "**/anaconda3": true,
    "**/.oh-my-zsh": true,
    "**/.gnupg": true,
    "**/.didim365": true,
    "**/.conda/**": true,
    "**/_Archive/**": true,
    "**/*.csv": true,
    "**/*.pickle": true
  },

  // other settings
  "workbench.list.openMode": "doubleClick",
  "workbench.editorAssociations": {
    "*.ipynb": "jupyter-notebook"
  }
}
```
