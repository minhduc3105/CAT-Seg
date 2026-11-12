#!/usr/bin/env bash
# Source this file from setup_env.sh so the selected compilers remain in scope.
set -euo pipefail

expected_cuda=11.8
cuda_dir=/usr/local/cuda-11.8

if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
  if [[ $(id -u) -ne 0 ]] || ! command -v apt-get >/dev/null 2>&1; then
    echo '[setup] GCC/G++ 11 are required. Install gcc-11 and g++-11, then rerun setup.' >&2
    exit 1
  fi
  echo '[setup] Installing GCC/G++ 11 for CUDA 11.8...'
  apt-get update
  apt-get install -y gcc-11 g++-11
fi

if [[ ! -x "$cuda_dir/bin/nvcc" ]]; then
  if [[ $(id -u) -ne 0 || $(uname -m) != x86_64 ]]; then
    echo '[setup] CUDA Toolkit 11.8 is missing. Install it at /usr/local/cuda-11.8, then rerun setup.' >&2
    exit 1
  fi
  command -v curl >/dev/null 2>&1 || {
    echo '[setup] curl is required to download CUDA Toolkit 11.8.' >&2; exit 1;
  }
  mkdir -p .deps
  installer=.deps/cuda_11.8.0_520.61.05_linux.run
  expected_md5=d6cf26349e5a4104b8683e26b717fa22
  if [[ ! -f "$installer" || $(stat -c %s "$installer") -ne 4336730777 ]]; then
    echo '[setup] Downloading CUDA Toolkit 11.8 (toolkit only)...'
    curl --fail --location --retry 3 --continue-at - --output "$installer" \
      'https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run'
  fi
  printf '%s  %s\n' "$expected_md5" "$installer" | md5sum --check --status || {
    echo '[setup] CUDA installer checksum does not match NVIDIA; remove the file and rerun setup.' >&2
    exit 1
  }
  echo '[setup] Installing CUDA Toolkit 11.8 (without a driver)...'
  sh "$installer" --silent --toolkit --toolkitpath="$cuda_dir" --override
fi

if [[ ! -x "$cuda_dir/bin/nvcc" ]] || ! "$cuda_dir/bin/nvcc" --version | grep -q 'release 11\.8,'; then
  echo "[setup] Expected CUDA Toolkit $expected_cuda at $cuda_dir; found an incompatible nvcc." >&2
  exit 1
fi

# The runfile needs several GB; release that space once installation is valid.
if [[ -n ${installer:-} ]]; then rm -f "$installer"; fi

export CUDA_HOME="$cuda_dir"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

echo "[setup] Build toolchain: $($CUDA_HOME/bin/nvcc --version | grep 'release ' | tail -1); $($CC -dumpfullversion); CUDA_HOME=$CUDA_HOME"
