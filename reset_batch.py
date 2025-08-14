# reset_batch.py
import argparse
import database_manager
from logger_config import log

def reset_backtest_batch(batch_id: str):
    """
    Safely resets a specific backtest batch by:
    1. Deleting all performance data associated with the batch.
    2. Setting the status of all predictions in the batch back to 'open'.
    """
    log.info(f"--- Preparing to reset backtest batch: '{batch_id}' ---")
    
    # Safety confirmation prompt
    confirm = input(f"Are you sure you want to delete all performance data and reset all predictions for batch '{batch_id}'? [y/n]: ")
    if confirm.lower() != 'y':
        log.warning("Operation cancelled by user.")
        return

    try:
        database_manager.init_db(purpose='scheduler')

        # 1. Delete performance records for the batch
        log.info(f"Deleting performance records for batch: {batch_id}...")
        perf_result = database_manager.performance_collection.delete_many({"batch_id": batch_id})
        log.info(f"Successfully deleted {perf_result.deleted_count} performance records.")

        # 2. Reset prediction statuses for the batch
        log.info(f"Resetting prediction statuses to 'open' for batch: {batch_id}...")
        pred_result = database_manager.predictions_collection.update_many(
            {"batch_id": batch_id},
            {"$set": {"status": "open"}}
        )
        log.info(f"Successfully reset {pred_result.modified_count} predictions.")
        
        log.info(f"--- ✅ Batch '{batch_id}' has been successfully reset. ---")

    except Exception as e:
        log.error(f"An error occurred during batch reset: {e}")
    finally:
        database_manager.close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset a backtest batch to its pre-run state.")
    parser.add_argument('--batch_id', required=True, help='The unique ID of the backtest batch to reset.')
    args = parser.parse_args()
    
    reset_backtest_batch(batch_id=args.batch_id)