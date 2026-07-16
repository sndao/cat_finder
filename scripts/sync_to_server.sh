#!/usr/bin/env bash

# ~/cat_finder/scripts/sync_to_server.sh

set -u

LOG_DIR="$HOME/var/log"
LOG_FILE="$LOG_DIR/sync_to_server.log"

SRC_DB="$HOME/data/databases/found_cats.db"
TMP_DB="$HOME/.tmp/found_cats_$$.db"

REMOTE_USER="ec2-user"
REMOTE_HOST="212.2.241.230"
REMOTE_DB="/home/ec2-user/data/databases/found_cats.db"

mkdir -p "$HOME/.tmp"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

{
    log "========== sync start =========="

    if ! command -v sqlite3 >/dev/null 2>&1; then
        log "ERROR: sqlite3 not found"
        exit 1
    fi

    if ! command -v scp >/dev/null 2>&1; then
        log "ERROR: scp not found"
        exit 1
    fi

    if [ ! -f "$SRC_DB" ]; then
        log "ERROR: source db missing: $SRC_DB"
        exit 1
    fi

    log "creating sqlite backup"
    sqlite3 "$SRC_DB" ".backup '$TMP_DB'"
    rc=$?

    if [ $rc -ne 0 ]; then
        log "ERROR: sqlite backup failed rc=$rc"
        exit $rc
    fi

    if [ ! -f "$TMP_DB" ]; then
        log "ERROR: backup file not created"
        exit 1
    fi

    size_bytes=$(wc -c < "$TMP_DB" | tr -d ' ')
    log "backup created: $TMP_DB ($size_bytes bytes)"

    log "verifying remote path"

    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout=15 \
        "$REMOTE_USER@$REMOTE_HOST" \
        "mkdir -p /home/ec2-user/data/databases"

    rc=$?

    if [ $rc -ne 0 ]; then
        log "ERROR: remote mkdir failed rc=$rc"
        exit $rc
    fi

    log "uploading database"

    scp \
        -o BatchMode=yes \
        -o ConnectTimeout=15 \
        "$TMP_DB" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DB"

    rc=$?

    if [ $rc -ne 0 ]; then
        log "ERROR: scp failed rc=$rc"
        exit $rc
    fi

    log "verifying remote file"

    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout=15 \
        "$REMOTE_USER@$REMOTE_HOST" \
        "ls -lh '$REMOTE_DB'"

    rc=$?

    if [ $rc -ne 0 ]; then
        log "ERROR: remote verification failed rc=$rc"
        exit $rc
    fi

    rm -f "$TMP_DB"

    log "temporary file removed"
    log "sync successful"
    log "========== sync end =========="

} >> "$LOG_FILE" 2>&1