#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/reproduce_ubuntu_linux.sh" "$@"
