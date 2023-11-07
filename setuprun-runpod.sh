#!/usr/bin/env bash

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# Install tk and python3.10-venv
echo "Installing tk and python3.10-venv..."
apt update -y && apt install -y python3-tk python3.10-venv

# Install required libcudnn release 8.7.0.84-1
echo "Installing required libcudnn release 8.7.0.84-1..."
apt install -y libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 --allow-change-held-packages

# Check if libssl is installed and install it if not
if ! dpkg -l | grep -q libssl1.1; then
    echo "Installing libssl..."
    wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
else
    echo "libssl already installed, skipping installation..."
fi

# Check if the venv folder doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating venv..."
    python3 -m venv "$SCRIPT_DIR/venv"

    # Activate the virtual environment
    echo "Activating venv..."
    source "$SCRIPT_DIR/venv/bin/activate" || exit 1

    # Run setup_linux.py script with platform requirements
    echo "Running setup_linux.py..."
    python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_runpod.txt --show_stdout --no_run_accelerate
    pip3 cache purge

    # Deactivate the virtual environment
    echo "Deactivating venv..."
    deactivate
else
    echo "venv already exists, skipping setup_linux.py..."
fi

# Configure accelerate
echo "Configuring accelerate..."
mkdir -p "/root/.cache/huggingface/accelerate"
cp "$SCRIPT_DIR/config_files/accelerate/runpod.yaml" "/root/.cache/huggingface/accelerate/default_config.yaml"

echo "Installation completed."

./gui.sh --headless --listen 0.0.0.0 --server_port 7878
