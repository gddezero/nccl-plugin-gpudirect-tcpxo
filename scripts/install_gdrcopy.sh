#!/bin/bash
# Install GDRCopy 2.4.4 on Rocky 9.7 + NVIDIA driver 580.
# Builds kernel module + userspace lib, loads kmod, registers libgdrapi.
# Idempotent: skips most work if /dev/gdrdrv exists and libgdrapi is loadable.
set -euo pipefail

PATH=/usr/sbin:/sbin:$PATH

if [ -c /dev/gdrdrv ] && ldconfig -p | grep -q libgdrapi; then
  echo "GDRCopy already installed and loaded; nothing to do."
  ls -l /dev/gdrdrv
  lsmod | grep gdrdrv
  exit 0
fi

KVER=$(uname -r)
dnf install -y kernel-devel-$KVER kernel-headers-$KVER \
  gcc make autoconf automake libtool dkms pkg-config

GDRCOPY_VER=2.4.4
WORKDIR=/usr/src/gdrcopy-$GDRCOPY_VER
if [ ! -d "$WORKDIR" ]; then
  curl -sL "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VER}.tar.gz" \
    -o /tmp/gdrcopy.tgz
  tar -C /usr/src -xzf /tmp/gdrcopy.tgz
fi

cd "$WORKDIR"

# 1. Userspace lib + headers
make prefix=/usr/local/gdrcopy CUDA=/usr/local/cuda lib lib_install

# 2. Kernel module (gdrdrv)
cd src/gdrdrv
make KVER=$KVER
INSTALL_DIR=/lib/modules/$KVER/extra
mkdir -p $INSTALL_DIR
cp gdrdrv.ko $INSTALL_DIR/
depmod -a

# 3. Load the module and create /dev/gdrdrv
modprobe gdrdrv || insmod $INSTALL_DIR/gdrdrv.ko
sleep 1
if [ ! -c /dev/gdrdrv ]; then
  # Create device node manually if udev didn't (kernel allocates a dynamic major)
  major=$(awk '$2=="gdrdrv"{print $1}' /proc/devices)
  if [ -n "$major" ]; then
    mknod -m 666 /dev/gdrdrv c "$major" 0
  fi
fi
chmod 666 /dev/gdrdrv

# 4. Persist module load across reboots
echo "gdrdrv" > /etc/modules-load.d/gdrdrv.conf

# 5. Register libgdrapi with ldconfig
echo /usr/local/gdrcopy/lib > /etc/ld.so.conf.d/gdrcopy.conf
ldconfig

echo "=== GDRCopy ${GDRCOPY_VER} installed ==="
ls -la /dev/gdrdrv
lsmod | grep gdrdrv
ldconfig -p | grep libgdrapi
