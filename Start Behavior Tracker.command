#!/bin/zsh
# Double-click to open the Behavior Tracker in your web browser.
cd "${0:A:h}"
if ! .venv/bin/python -c "import sys" 2>/dev/null; then
  echo "First-time setup (a few minutes)…"
  rm -rf .venv
  python3 -m venv .venv || { echo "Setup failed: is Python 3 installed?"; read; exit 1; }
fi
if ! .venv/bin/python -c "import streamlit, cv2, skimage, tifffile" 2>/dev/null; then
  echo "Installing what the app needs (a few minutes)…"
  .venv/bin/pip install -q -r requirements.txt || { echo "Setup failed"; read; exit 1; }
fi
echo "Opening Behavior Tracker… (close this window to stop it)"
.venv/bin/python -m streamlit run app.py --server.headless false
