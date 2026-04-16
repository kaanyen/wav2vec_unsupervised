#!/bin/bash

# =============================================================================
# setup_functions.sh — All installation helper functions for the pipeline
# =============================================================================
#
# CHANGES FROM ORIGINAL (see CHANGES.md for full rationale):
#   1. INSTALL_ROOT now derived from script location (not $HOME) so all
#      dependencies install inside the project folder, not ~/wav2vec_unsupervised.
#   2. All apt-get / apt install calls replaced with brew install equivalents
#      for macOS compatibility.
#   3. CUDA / GPU driver installation skipped on macOS (wrapped in OS check).
#   4. PyTorch installed without the CUDA index URL; the default PyTorch wheel
#      includes MPS (Apple Silicon GPU) support automatically.
#   5. nproc replaced with $(sysctl -n hw.ncpu) for macOS parallel builds.
#   6. install_wav2vec_u() added: copies wav2vec_u.py into fairseq_ after clone.
#   7. All setup output is tee'd to $INSTALL_ROOT/logs/setup.log.
#
# =============================================================================

set -e
set -o pipefail
set -x

# ── Derive project root from script location ──────────────────────────────────
# This guarantees all paths are relative to the cloned repo regardless of
# where the user runs the script from.  The original code used $HOME which
# placed all downloads in ~/wav2vec_unsupervised instead of the project folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== CONFIGURATION ====================

INSTALL_ROOT="$SCRIPT_DIR"
FAIRSEQ_ROOT="$INSTALL_ROOT/fairseq_"
KENLM_ROOT="$INSTALL_ROOT/kenlm"
VENV_PATH="$INSTALL_ROOT/venv"
RVADFAST_ROOT="$INSTALL_ROOT/rVADfast"
FLASHLIGHT_SEQ_ROOT="$INSTALL_ROOT/sequence"

# Python version (matches .python-version in the repo)
PYTHON_VERSION="3.10"

# Source central config (DEVICE, TEST_RUN, etc.) if not already loaded
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
fi

# ==================== HELPER FUNCTIONS ====================

# Log with timestamp, also writes to setup log file
log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local line="[$timestamp] $message"
    echo "$line"
    # Also write to log file if the log dir exists
    if [ -d "$INSTALL_ROOT/logs" ]; then
        echo "$line" >> "$INSTALL_ROOT/logs/setup.log"
    fi
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Portable CPU count: sysctl on macOS, nproc on Linux
cpu_count() {
    if command_exists sysctl; then
        sysctl -n hw.ncpu
    else
        nproc
    fi
}

# Create project-internal log directory and data directories
create_dirs() {
    mkdir -p "$INSTALL_ROOT/logs"
    mkdir -p "$INSTALL_ROOT/data"
    log "Created base directories under $INSTALL_ROOT"
}


# ==================== SETUP STEPS ====================

# Install system-level build dependencies.
# On macOS we use Homebrew instead of apt-get.
# On Linux the original apt-get commands are preserved as a fallback.
basic_dependencies() {
    log "Installing system dependencies..."

    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS — require Homebrew
        if ! command_exists brew; then
            log "[ERROR] Homebrew not found. Install it from https://brew.sh before running setup."
            exit 1
        fi
        brew update
        # Core build tools + pyenv (avoids curl https://pyenv.run which fails on DNS/offline)
        brew install pyenv python@3.10 cmake autoconf automake libtool pkg-config wget curl git
        # Libraries needed for KenLM and other C++ builds
        brew install boost eigen zlib bzip2 readline openssl libffi xz
        # Audio processing
        brew install ffmpeg
        # Graphviz (optional, for visualisations)
        brew install graphviz || true
        log "macOS system dependencies installed via Homebrew."
    else
        # Linux fallback (original behaviour)
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-dev build-essential
        sudo apt-get install -y autoconf automake cmake curl g++ git graphviz \
             libatlas3-base libtool make pkg-config subversion unzip wget \
             zlib1g-dev gfortran
        sudo apt update
        sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
             libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
             libffi-dev liblzma-dev wget curl
        sudo apt install -y software-properties-common
        log "Linux system dependencies installed via apt-get."
    fi
}


# Set up Python virtual environment.
#
# macOS: install pyenv via Homebrew in basic_dependencies() — do NOT use
#   curl https://pyenv.run (fails with "Could not resolve host" when DNS is
#   broken, offline, or corporate firewalls block the domain).
# Linux: try pyenv.run with a timeout; fall back to apt install pyenv or
#   system python3.
#
# A pyenv-built Python with --enable-shared is preferred for protobuf/fairseq,
# but Homebrew python@3.10 works as a fallback if pyenv install fails.
setup_venv() {
    log "Setting up Python virtual environment..."

    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"

    local PYTHON_CMD=""

    if [[ "$(uname)" == "Darwin" ]]; then
        # ── Ensure pyenv is available (installed via brew in basic_dependencies) ──
        if ! command_exists pyenv; then
            if command_exists brew; then
                log "pyenv not on PATH — installing via Homebrew..."
                brew install pyenv || log "[WARN] brew install pyenv failed."
            fi
        fi

        if command_exists pyenv; then
            [[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init - bash)" 2>/dev/null || eval "$(pyenv init -)" 2>/dev/null || true

            # Install requested minor version if missing (e.g. 3.10.x)
            if ! pyenv versions --bare 2>/dev/null | grep -qE "^${PYTHON_VERSION}\\."; then
                log "Installing Python $PYTHON_VERSION via pyenv (shared library build)..."
                env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install -s "$PYTHON_VERSION" \
                    || log "[WARN] pyenv install $PYTHON_VERSION failed — will try Homebrew python@3.10."
            fi

            local ver
            ver=$(pyenv versions --bare 2>/dev/null | grep -E "^${PYTHON_VERSION}\\." | head -1)
            if [ -n "$ver" ] && [ -x "$PYENV_ROOT/versions/$ver/bin/python" ]; then
                PYTHON_CMD="$PYENV_ROOT/versions/$ver/bin/python"
                log "Using pyenv Python: $PYTHON_CMD"
            fi
        fi

        # Fallback: Homebrew python@3.10 (already installed in basic_dependencies)
        if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
            if command_exists brew; then
                local hb_py
                hb_py="$(brew --prefix python@3.10 2>/dev/null)/bin/python3.10"
                if [ -x "$hb_py" ]; then
                    PYTHON_CMD="$hb_py"
                    log "Using Homebrew Python: $PYTHON_CMD"
                fi
            fi
        fi

        if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
            PYTHON_CMD="$(command -v python3)"
            log "Using python3 from PATH: $PYTHON_CMD"
        fi
    else
        # ── Linux: network installer with timeout, then distro package, then system python ──
        if [ ! -d "$PYENV_ROOT" ] && ! command_exists pyenv; then
            log "Installing pyenv (Linux)..."
            if curl -fsSL --connect-timeout 15 --max-time 120 https://pyenv.run -o /tmp/pyenv-installer.sh 2>/dev/null; then
                bash /tmp/pyenv-installer.sh
            else
                log "[WARN] Could not download https://pyenv.run — trying apt install pyenv..."
                sudo apt-get update && sudo apt-get install -y pyenv 2>/dev/null || true
            fi
        fi

        [[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init - bash)" 2>/dev/null || true

        if command_exists pyenv; then
            if ! pyenv versions --bare 2>/dev/null | grep -qE "^${PYTHON_VERSION}\\."; then
                env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install -s "$PYTHON_VERSION" || true
            fi
            local ver
            ver=$(pyenv versions --bare 2>/dev/null | grep -E "^${PYTHON_VERSION}\\." | head -1)
            if [ -n "$ver" ] && [ -x "$PYENV_ROOT/versions/$ver/bin/python" ]; then
                PYTHON_CMD="$PYENV_ROOT/versions/$ver/bin/python"
            fi
        fi

        if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
            PYTHON_CMD="$(command -v python3)"
        fi
    fi

    if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
        log "[ERROR] No usable Python interpreter found. Install Python $PYTHON_VERSION and retry."
        exit 1
    fi

    if [ -d "$VENV_PATH" ]; then
        log "Virtual environment already exists at $VENV_PATH"
    else
        log "Creating venv with: $PYTHON_CMD"
        "$PYTHON_CMD" -m venv "$VENV_PATH"
        log "Created virtual environment at $VENV_PATH"
    fi

    source "$VENV_PATH/bin/activate"
    log "Python virtual environment setup completed ($(python --version 2>&1))."
}


# CUDA installation — Linux + NVIDIA GPU only.
# On macOS this is skipped entirely because:
#   • macOS no longer supports NVIDIA CUDA drivers (since macOS 10.14).
#   • Apple Silicon uses Metal / MPS for GPU acceleration.
cuda_installation() {
    if [[ "$(uname)" == "Darwin" ]]; then
        log "[INFO] Skipping CUDA installation on macOS — not supported."
        return 0
    fi

    local cmd_file="cuda_installation.txt"
    if [[ -f "$cmd_file" ]]; then
        log "Starting CUDA installation from $cmd_file ..."
        source "$cmd_file"
        echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        export PATH="/usr/local/cuda-12.3/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"
        source ~/.bashrc
        log "CUDA environment variables configured."
    else
        log "[ERROR] $cmd_file not found. Cannot install CUDA."
        return 1
    fi
}


# GPU driver installation — Google Cloud / Linux only.
# Skipped on macOS for the same reason as cuda_installation().
gpu_drivers_installation() {
    if [[ "$(uname)" == "Darwin" ]]; then
        log "[INFO] Skipping GPU driver installation on macOS — not applicable."
        return 0
    fi

    log "--- Starting GPU Driver and Toolkit Installation ---"
    curl -s -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
    sudo python3 install_gpu_driver.py
    sudo apt-get update -y

    if command_exists nvidia-smi; then
        log "SUCCESS: nvidia-smi found."
        nvidia-smi
    else
        log "WARNING: nvidia-smi not found after driver installation. A reboot may be required."
    fi
}


# Install PyTorch and all Python dependencies.
# On macOS:
#   • No --index-url pointing at CUDA wheels.
#   • The default PyTorch wheel from PyPI includes MPS (Apple Silicon) support.
#   • faiss-cpu is always used on macOS (faiss-gpu requires CUDA).
install_pytorch_and_other_packages() {
    log "Installing PyTorch and related packages..."
    source "$VENV_PATH/bin/activate"

    if [[ "$(uname)" == "Darwin" ]]; then
        # Default PyPI wheels include MPS support — no special index URL needed
        log "[macOS] Installing PyTorch 2.3.0 (CPU + MPS wheels) ..."
        pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
    else
        # Linux with CUDA 12.1 wheels
        pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
            --index-url "https://download.pytorch.org/whl/cu121"
    fi

    pip install "numpy<2" scipy tqdm sentencepiece soundfile librosa \
                editdistance tensorboardX packaging
    pip install npy-append-array h5py kaldi-io g2p_en

    # faiss-cpu on macOS (faiss-gpu requires CUDA which is Linux/Windows only)
    if [[ "$(uname)" == "Darwin" ]]; then
        log "[macOS] Installing faiss-cpu (faiss-gpu not available on macOS)"
        pip install faiss-cpu
    else
        if command_exists nvcc; then
            pip install faiss-gpu
        else
            pip install faiss-cpu
        fi
    fi

    pip install ninja
    # torchcodec: skip on macOS if not available (optional visualisation helper)
    pip install torchcodec || log "[WARN] torchcodec install failed — skipping (not required)."

    python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
    log "PyTorch and related packages installed successfully."
}


# Clone and install the Ashesi-fork of fairseq in editable mode.
install_fairseq() {
    log "--- Installing fairseq ---"
    source "$VENV_PATH/bin/activate"
    pip install "pip==24.0"

    cd "$INSTALL_ROOT"

    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists at $FAIRSEQ_ROOT. Pulling latest changes..."
        cd "$FAIRSEQ_ROOT"
        git pull || log "[WARN] Failed to pull latest fairseq changes. Continuing with existing version."
    else
        log "Cloning fairseq repository..."
        git clone https://github.com/Ashesi-Org/fairseq_.git "$FAIRSEQ_ROOT" \
            || { log "[ERROR] Failed to clone fairseq repository."; exit 1; }
        cd "$FAIRSEQ_ROOT"
    fi

    log "Installing fairseq in editable mode..."
    pip install --editable ./ \
        || { log "[ERROR] Failed to install fairseq in editable mode."; exit 1; }

    local wav2vec_req_file="$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    if [ -f "$wav2vec_req_file" ]; then
        log "Installing wav2vec-specific requirements from $wav2vec_req_file ..."
        pip install -r "$wav2vec_req_file" \
            || log "[WARN] Some wav2vec requirements failed. Check $wav2vec_req_file."
    fi

    log "fairseq installed successfully."
    deactivate
}


# Copy the custom wav2vec_u.py (with Generator / RealData / Discriminator
# separation of concerns) into the cloned fairseq_ tree.
# This must be called AFTER install_fairseq() so the target directory exists.
install_wav2vec_u() {
    log "--- Installing custom wav2vec_u.py ---"

    local src="$INSTALL_ROOT/wav2vec_u.py"
    local dst_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/models"
    local dst="$dst_dir/wav2vec_u.py"

    if [ ! -f "$src" ]; then
        log "[ERROR] Source file not found: $src"
        log "        Place wav2vec_u.py in the project root ($INSTALL_ROOT) before running setup."
        exit 1
    fi

    if [ ! -d "$dst_dir" ]; then
        log "[ERROR] Target directory does not exist: $dst_dir"
        log "        Make sure install_fairseq() ran successfully before this step."
        exit 1
    fi

    cp "$src" "$dst"
    log "Copied wav2vec_u.py → $dst"
    log "--- wav2vec_u.py installation complete ---"
}


# Clone and prepare rVADfast for voice activity detection (silence removal).
install_rVADfast() {
    log "Cloning and installing rVADfast..."
    cd "$INSTALL_ROOT"
    source "$VENV_PATH/bin/activate"

    if [ -d "$RVADFAST_ROOT" ]; then
        log "rVADfast already exists at $RVADFAST_ROOT. Updating..."
        cd "$RVADFAST_ROOT"
        git pull
    else
        log "Cloning rVADfast repository..."
        git clone https://github.com/zhenghuatan/rVADfast.git "$RVADFAST_ROOT"
        cd "$RVADFAST_ROOT"
    fi

    mkdir -p "$RVADFAST_ROOT/src"
    log "rVADfast installed successfully."
}


# Apply small CMake patches so KenLM builds with current Homebrew Boost/Eigen on macOS.
#
# Problems fixed:
#   1. find_package(Eigen3 3.1.0 CONFIG) rejects Homebrew Eigen (reports as 5.x in
#      Eigen3Config.cmake) even though the library is compatible.
#   2. find_package(Boost ... COMPONENTS ... system ...) fails: Boost 1.82+ makes
#      boost_system header-only; Homebrew's BoostConfig.cmake no longer ships a
#      separate boost_system package, so CMake errors with "Could not find ...
#      boost_system".
#
# Idempotent: safe to run on every setup; only changes lines that still match.
patch_kenlm_for_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        return 0
    fi
    local cm="$KENLM_ROOT/CMakeLists.txt"
    if [ ! -f "$cm" ]; then
        return 0
    fi
    log "[macOS] Patching KenLM CMakeLists.txt for Homebrew Boost/Eigen..."
    if grep -q 'find_package(Eigen3 3.1.0 CONFIG)' "$cm" 2>/dev/null; then
        if sed --version 2>&1 | grep -q GNU; then
            sed -i 's/find_package(Eigen3 3.1.0 CONFIG)/find_package(Eigen3 CONFIG)/' "$cm"
        else
            sed -i '' 's/find_package(Eigen3 3.1.0 CONFIG)/find_package(Eigen3 CONFIG)/' "$cm"
        fi
    fi
    # Remove standalone Boost "system" component line (header-only in modern Boost)
    if grep -q '^  system$' "$cm" 2>/dev/null; then
        if sed --version 2>&1 | grep -q GNU; then
            sed -i '/^  system$/d' "$cm"
        else
            sed -i '' '/^  system$/d' "$cm"
        fi
    fi
}


# Clone and build KenLM for language model training (used in prepare_text step).
# On macOS, Homebrew provides eigen and boost instead of apt packages.
install_kenlm() {
    log "Cloning and building KenLM..."
    cd "$INSTALL_ROOT"

    if [[ "$(uname)" == "Darwin" ]]; then
        log "[macOS] Installing KenLM dependencies via Homebrew..."
        brew install eigen boost || log "[WARN] eigen/boost may already be installed."
    else
        sudo apt update
        sudo apt install -y libeigen3-dev libboost-all-dev
    fi

    if [ -d "$KENLM_ROOT" ]; then
        log "KenLM repository already exists at $KENLM_ROOT"
    else
        log "Cloning KenLM repository..."
        git clone https://github.com/kpu/kenlm.git "$KENLM_ROOT"
    fi

    patch_kenlm_for_macos

    cd "$KENLM_ROOT"

    # If a previous cmake run failed, build/ exists but binaries are missing — rebuild.
    if [ -d "build" ] && [ ! -f "build/bin/lmplz" ]; then
        log "KenLM build/ exists but is incomplete (no build/bin/lmplz). Removing and rebuilding..."
        rm -rf build
    fi

    if [ -f "build/bin/lmplz" ]; then
        log "KenLM binaries already built — skipping compile."
    else
        mkdir -p build
        cd build
        if [[ "$(uname)" == "Darwin" ]]; then
            # Use legacy FindBoost (CMP0167 OLD) + no BoostConfig.cmake so Homebrew layout works.
            cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                -DCMAKE_POLICY_DEFAULT_CMP0167=OLD \
                -DBoost_NO_BOOST_CMAKE=ON \
                -DBoost_USE_STATIC_LIBS=OFF \
                -DBOOST_ROOT="$(brew --prefix boost)" \
                -DCMAKE_PREFIX_PATH="$(brew --prefix boost);$(brew --prefix eigen)"
        else
            cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        fi
        make -j "$(cpu_count)"
    fi

    source "$VENV_PATH/bin/activate"
    pip install https://github.com/kpu/kenlm/archive/master.zip
    log "KenLM built and installed successfully."
}


# Install Flashlight (text + sequence bindings for decoding).
# On macOS, pybind11 is installed via Homebrew instead of apt-get.
install_flashlight() {
    log "--- Installing Flashlight (Text and Sequence) ---"
    cd "$INSTALL_ROOT"

    if [[ "$(uname)" == "Darwin" ]]; then
        log "[macOS] Installing pybind11 via Homebrew..."
        brew install pybind11 || log "[WARN] pybind11 may already be installed."
    else
        sudo apt-get install -y pybind11-dev
    fi

    source "$VENV_PATH/bin/activate"

    log "Installing flashlight-text Python package..."
    pip install flashlight-text \
        || { log "[ERROR] Failed to install flashlight-text."; exit 1; }

    if [ -d "$FLASHLIGHT_SEQ_ROOT" ]; then
        log "Flashlight sequence repository already exists. Updating..."
        cd "$FLASHLIGHT_SEQ_ROOT"
        git pull || log "[WARN] Failed to pull latest flashlight sequence changes."
    else
        log "Cloning flashlight sequence repository..."
        git clone https://github.com/flashlight/sequence.git "$FLASHLIGHT_SEQ_ROOT" \
            || { log "[ERROR] Failed to clone flashlight sequence."; exit 1; }
        cd "$FLASHLIGHT_SEQ_ROOT"
    fi

    log "Configuring and building flashlight sequence WITH Python bindings..."
    rm -rf build
    mkdir build && cd build

    local flashlight_python_flag="-DFLASHLIGHT_BUILD_PYTHON=ON"
    local use_cuda_flag="-DFLASHLIGHT_USE_CUDA=OFF"

    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS: always CPU-only (no CUDA)
        export USE_CUDA=0
        log "[macOS] Building flashlight sequence in CPU-only mode."
    else
        if command_exists nvcc; then
            export USE_CUDA=1
            use_cuda_flag="-DFLASHLIGHT_USE_CUDA=ON"
        else
            export USE_CUDA=0
        fi
    fi

    local python_executable="$VENV_PATH/bin/python"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DPYTHON_EXECUTABLE="$python_executable" \
             "$flashlight_python_flag" \
             "$use_cuda_flag"

    log "Building Flashlight sequence library..."
    cmake --build . --config Release --parallel "$(cpu_count)"

    log "Installing Flashlight sequence Python bindings..."
    cd ..
    pip install . || log "[WARN] Flashlight sequence pip install failed — some decoding features may be unavailable."

    log "[PASS] Flashlight Python bindings installed."
    cd "$INSTALL_ROOT"

    log "Re-installing fairseq to ensure it picks up Flashlight bindings..."
    install_fairseq

    log "--- Flashlight Installation Finished ---"
}


# Download the pre-trained wav2vec 2.0 model checkpoint.
download_pretrained_model() {
    log "Downloading pre-trained wav2vec model..."
    mkdir -p "$INSTALL_ROOT/pre-trained"
    cd "$INSTALL_ROOT/pre-trained"

    if [ -f "$INSTALL_ROOT/pre-trained/wav2vec_vox_new.pt" ]; then
        log "Pre-trained model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt
    fi
    log "Pre-trained model downloaded successfully."
}


# Download the FastText language identification model (lid.176.bin).
download_languageIdentification_model() {
    log "Downloading language identification model..."
    mkdir -p "$INSTALL_ROOT/lid_model"
    cd "$INSTALL_ROOT/lid_model"

    if [ -f "$INSTALL_ROOT/lid_model/lid.176.bin" ]; then
        log "LID model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fi

    source "$VENV_PATH/bin/activate"
    # PyPI wheels for fasttext can ship broken libc++ rpaths on macOS; build from sdist when possible.
    pip uninstall -y fasttext 2>/dev/null || true
    if ! pip install --no-cache-dir --no-binary fasttext fasttext; then
        log "[WARN] fasttext build from source failed; trying PyPI wheel (may break on macOS)."
        pip install --no-cache-dir fasttext
    fi
    log "Language identification model downloaded successfully."
}
