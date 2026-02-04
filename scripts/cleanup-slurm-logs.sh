#!/bin/bash
#
# Slurm Test Log Cleanup Script
#
# Automatically cleans up old test result logs to prevent disk space issues.
# Recommended to run via cron daily.
#
# Usage:
#   ./cleanup-slurm-logs.sh [results-dir] [retention-days]
#
# Examples:
#   ./cleanup-slurm-logs.sh ./results 7     # Keep last 7 days
#   ./cleanup-slurm-logs.sh ./results 30    # Keep last 30 days
#
# Crontab example (daily at 2 AM):
#   0 2 * * * /path/to/cleanup-slurm-logs.sh /path/to/results 7 >> /var/log/cleanup-tests.log 2>&1
#

set -e

RESULTS_DIR=${1:-./results}
RETENTION_DAYS=${2:-7}
DRY_RUN=${DRY_RUN:-false}

echo "=========================================="
echo "Slurm Test Log Cleanup Script"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Retention period: $RETENTION_DAYS days"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' not found"
    exit 1
fi

# Report current disk usage
echo "Current disk usage:"
du -sh "$RESULTS_DIR"
echo ""

# Count files by age
TOTAL_FILES=$(find "$RESULTS_DIR" -name "*.out" -type f 2>/dev/null | wc -l)
OLD_FILES=$(find "$RESULTS_DIR" -name "*.out" -type f -mtime +${RETENTION_DAYS} 2>/dev/null | wc -l)
RECENT_FILES=$((TOTAL_FILES - OLD_FILES))

echo "File summary:"
echo "  - Total log files: $TOTAL_FILES"
echo "  - Files to keep (< $RETENTION_DAYS days): $RECENT_FILES"
echo "  - Files to delete (> $RETENTION_DAYS days): $OLD_FILES"
echo ""

if [ "$OLD_FILES" -eq 0 ]; then
    echo "✓ No files to clean up"
    exit 0
fi

# List files to be deleted
echo "Files to be deleted:"
find "$RESULTS_DIR" -name "*.out" -type f -mtime +${RETENTION_DAYS} 2>/dev/null | while read -r file; do
    age_days=$((($(date +%s) - $(stat -f %m "$file" 2>/dev/null || stat -c %Y "$file")) / 86400))
    size=$(du -h "$file" | cut -f1)
    echo "  - $(basename "$file") (age: ${age_days}d, size: $size)"
done
echo ""

# Delete old files
if [ "$DRY_RUN" = "false" ]; then
    echo "Deleting files older than $RETENTION_DAYS days..."
    DELETED_COUNT=$(find "$RESULTS_DIR" -name "*.out" -type f -mtime +${RETENTION_DAYS} -delete -print 2>/dev/null | wc -l)
    echo "✓ Deleted $DELETED_COUNT files"
else
    echo "(Dry run - no files deleted)"
fi

echo ""

# Report new disk usage
echo "=========================================="
echo "Disk usage after cleanup:"
du -sh "$RESULTS_DIR"
echo "=========================================="

# Optional: Archive old logs before deletion
# Uncomment the following section to enable archiving

# ARCHIVE_DIR="/path/to/archive"
# if [ "$OLD_FILES" -gt 0 ] && [ -d "$ARCHIVE_DIR" ]; then
#     ARCHIVE_NAME="test-logs-$(date +%Y%m%d-%H%M%S).tar.gz"
#     echo ""
#     echo "Archiving old logs to $ARCHIVE_DIR/$ARCHIVE_NAME..."
#     find "$RESULTS_DIR" -name "*.out" -type f -mtime +${RETENTION_DAYS} -print0 | \
#         tar -czf "$ARCHIVE_DIR/$ARCHIVE_NAME" --null -T -
#     echo "✓ Archive created"
# fi

echo ""
echo "Cleanup complete!"
