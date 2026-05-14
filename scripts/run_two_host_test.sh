#!/bin/bash
# Cross-host two-rank GIN PROXY test orchestrator (script-file approach).
set -e
H1=forrest-h100-01
H2=forrest-h100-02
USR=maxwellx_google_com
UID_FILE=/tmp/nccl_uid_$$

# 1. Build the in-container runner script and stage it on both hosts/containers.
cat > /tmp/inner_runner.sh <<'EOF'
#!/bin/bash
set -e
unset LD_PRELOAD
# Bypass the /var/lib/tcpxo/lib64/libnccl-net-shim guest config checker;
# point NCCL straight at the actual FasTrak v7 plugin so its env-var
# whitelist does not reject NCCL_GIN_PLUGIN.
export NCCL_NET_PLUGIN=/plugins/libnccl-net.so
export NCCL_TUNER_CONFIG_PATH=/plugins/a3plus_tuner_config.textproto
export LD_LIBRARY_PATH=/tmp:$LD_LIBRARY_PATH
export NCCL_GIN_PLUGIN=/tmp/libnccl-gin.so
export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_FASTRAK_CTRL_DEV=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_FASTRAK_USE_LLCM=1
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GIN,BOOTSTRAP,ENV
echo "[runner] env NCCL_GIN_PLUGIN=$NCCL_GIN_PLUGIN" 1>&2
echo "[runner] env NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN" 1>&2
echo "[runner] env NCCL_TUNER_PLUGIN=$NCCL_TUNER_PLUGIN" 1>&2
exec /tmp/two_host_nccl_test "$@"
EOF
chmod +x /tmp/inner_runner.sh

for H in ${H1} ${H2}; do
  scp /tmp/inner_runner.sh ${USR}@${H}:/tmp/inner_runner.sh >/dev/null
  ssh ${USR}@${H} "sg docker -c 'docker cp /tmp/inner_runner.sh fastrakTestContainer:/tmp/inner_runner.sh && docker exec fastrakTestContainer chmod +x /tmp/inner_runner.sh && docker exec fastrakTestContainer rm -f /tmp/nccl_uid_*'" >/dev/null
done

# 2. Launch rank 0 on h100-01.
echo "==> launching rank 0 on ${H1}"
( ssh ${USR}@${H1} "sg docker -c 'docker exec fastrakTestContainer /tmp/inner_runner.sh 0 2 ${UID_FILE}'" > /tmp/rank0.log 2>&1 ) &
PID0=$!

# 3. Wait for the UID file then ferry to h100-02.
echo "==> waiting for uniqueId file on ${H1}"
for i in $(seq 1 60); do
  if ssh ${USR}@${H1} "sg docker -c 'docker exec fastrakTestContainer ls ${UID_FILE} 2>/dev/null'" 2>/dev/null | grep -q "${UID_FILE}"; then
    echo "==> uniqueId ready, ferrying"
    ssh ${USR}@${H1} "sg docker -c 'docker cp fastrakTestContainer:${UID_FILE} ${UID_FILE}'"
    scp ${USR}@${H1}:${UID_FILE} ${UID_FILE}
    scp ${UID_FILE} ${USR}@${H2}:${UID_FILE}
    ssh ${USR}@${H2} "sg docker -c 'docker cp ${UID_FILE} fastrakTestContainer:${UID_FILE}'"
    break
  fi
  sleep 1
done

# 4. Launch rank 1 on h100-02.
echo "==> launching rank 1 on ${H2}"
( ssh ${USR}@${H2} "sg docker -c 'docker exec fastrakTestContainer /tmp/inner_runner.sh 1 2 ${UID_FILE}'" > /tmp/rank1.log 2>&1 ) &
PID1=$!

wait ${PID0}; rc0=$?
wait ${PID1}; rc1=$?
echo "==> rank 0 exit ${rc0}, rank 1 exit ${rc1}"
echo "--- rank 0 last 30 lines ---"; tail -30 /tmp/rank0.log
echo "--- rank 1 last 30 lines ---"; tail -30 /tmp/rank1.log
