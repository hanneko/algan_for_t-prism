USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

usermod -u $USER_ID vscode
groupmod -g $GROUP_ID vscode