import subprocess
from datetime import datetime

# Configurations
MODELS = ["coxph", "coxnet", "rsf", "rsf_tuned", "xgboost"]
IMPUTATIONS = ["zero", "knn", "knn_tuned", "mice"]

# Base command including the --shap flag
BASE_COMMAND = ["python", "-m", "src.baseline.main_baseline", "--shap"]


def run_experiments():
    start_time = datetime.now()
    log_file = f"batch_run_log_{start_time.strftime('%Y%m%d_%H%M')}.txt"

    def log_and_print(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log_and_print(
        f"=== Batch Job Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
    )

    total = len(MODELS) * len(IMPUTATIONS)
    count = 0

    for model in MODELS:
        for imp in IMPUTATIONS:
            count += 1
            cmd = BASE_COMMAND + ["--model", model, "--imputation", imp]

            msg_start = f"\n[{count}/{total}] Executing: {' '.join(cmd)} (Start: {datetime.now().strftime('%H:%M:%S')})"
            log_and_print(msg_start)

            try:
                # Execute the command (output streams directly to the terminal)
                subprocess.run(cmd, check=True)
                msg_ok = f" -> SUCCESS: Combination {model} + {imp} completed."
                log_and_print(msg_ok)
            except subprocess.CalledProcessError:
                msg_err = (
                    f" -> ERROR: Combination {model} + {imp} failed. Skipping to next."
                )
                log_and_print(msg_err)
                continue

    end_time = datetime.now()
    duration = end_time - start_time
    log_and_print(f"\n{'=' * 60}")
    log_and_print(f"=== Batch Job End: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_and_print(f"Total duration: {duration}")
    log_and_print(f"{'=' * 60}")


if __name__ == "__main__":
    run_experiments()
