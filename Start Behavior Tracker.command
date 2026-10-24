#!/bin/zsh
# Double-click to open the Behavior Tracker in your web browser.
cd "${0:A:h}"
if ! .venv/bin/python -c "import sys" 2>/dev/null; then
  echo "First-time setup (a few minutes)…"
  rm -rf .venv .venv.nosync
  # The libraries live in .venv.nosync so iCloud Drive won't sync (or hide) them;
  # .venv is a shortcut to it, so the usual .venv paths still work.
  python3 -m venv .venv.nosync && ln -s .venv.nosync .venv \
    || { echo "Setup failed: is Python 3 installed?"; read; exit 1; }
fi
chflags -R nohidden .venv.nosync 2>/dev/null  # Python ignores install files marked hidden
# (re)install when something is missing, e.g. after an update added a library
if ! .venv/bin/python -c "import importlib.metadata as m; m.version('behavetrack'); import streamlit, cv2, skimage, tifffile" 2>/dev/null; then
  echo "Installing what the app needs (a few minutes)…"
  .venv/bin/pip install -q -e ".[ai]" || { echo "Setup failed"; read; exit 1; }
fi
echo "Opening Behavior Tracker… (close this window to stop it)"
.venv/bin/python -m behavetrack app
