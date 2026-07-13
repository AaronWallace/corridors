#!/usr/bin/env bash
set -euo pipefail

# GitHub authentication setup for a fresh Ubuntu compute pod.
#
# Interactive:
#   ./gh_setup.sh
#
# Noninteractive (inject GH_TOKEN from the pod's secret manager):
#   GIT_USER_NAME="Aaron Wallace" GIT_USER_EMAIL="you@example.com" \
#   GH_TOKEN="..." ./gh_setup.sh

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup script is intended for Linux hosts." >&2
    exit 1
fi

install_packages() {
    local missing=()
    command -v git >/dev/null 2>&1 || missing+=(git)
    command -v gh >/dev/null 2>&1 || missing+=(gh)
    if ((${#missing[@]} == 0)); then
        return
    fi

    echo "Installing: ${missing[*]}"
    if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
        sudo apt-get update
        sudo apt-get install -y "${missing[@]}"
    else
        apt-get update
        apt-get install -y "${missing[@]}"
    fi
}

prompt_identity() {
    local current_name current_email
    current_name="${GIT_USER_NAME:-$(git config --global user.name || true)}"
    current_email="${GIT_USER_EMAIL:-$(git config --global user.email || true)}"

    if [[ -z "$current_name" ]]; then
        if [[ ! -t 0 ]]; then
            echo "Set GIT_USER_NAME for noninteractive setup." >&2
            exit 1
        fi
        read -r -p "Git commit name: " current_name
    fi
    if [[ -z "$current_email" ]]; then
        if [[ ! -t 0 ]]; then
            echo "Set GIT_USER_EMAIL for noninteractive setup." >&2
            exit 1
        fi
        read -r -p "GitHub commit email: " current_email
    fi
    if [[ -z "$current_name" || -z "$current_email" ]]; then
        echo "Git name and email cannot be empty." >&2
        exit 1
    fi

    git config --global user.name "$current_name"
    git config --global user.email "$current_email"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
}

authenticate() {
    if [[ -n "${GH_TOKEN:-}" ]]; then
        echo "Authenticating with GH_TOKEN from the environment..."
        # gh automatically reads GH_TOKEN; this installs its Git credential helper.
        gh auth setup-git --hostname github.com
    elif gh auth status --hostname github.com >/dev/null 2>&1; then
        echo "GitHub CLI is already authenticated."
        gh auth setup-git --hostname github.com
    else
        if [[ ! -t 0 ]]; then
            echo "Set GH_TOKEN or run this script interactively." >&2
            exit 1
        fi
        echo "Starting GitHub's browser/device-code login..."
        gh auth login --hostname github.com --git-protocol https --web
        gh auth setup-git --hostname github.com
    fi
}

normalize_remote() {
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return
    fi
    local remote
    remote="$(git remote get-url origin 2>/dev/null || true)"
    case "$remote" in
        git@github.com:*)
            remote="https://github.com/${remote#git@github.com:}"
            git remote set-url origin "$remote"
            echo "Updated origin to HTTPS: $remote"
            ;;
        ssh://git@github.com/*)
            remote="https://github.com/${remote#ssh://git@github.com/}"
            git remote set-url origin "$remote"
            echo "Updated origin to HTTPS: $remote"
            ;;
    esac
}

install_packages
prompt_identity
authenticate
normalize_remote

echo
gh auth status --hostname github.com
echo
echo "GitHub setup complete. You can now use git pull and git push."
