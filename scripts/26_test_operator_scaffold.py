import subprocess
import time
import sys

def run_command(cmd, wait=True):
    print(f"Running: {cmd}")
    if wait:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout.strip()
    else:
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def main():
    print("--- Test Operator Scaffold ---")
    
    # 1. Apply CRD
    print("Applying CRD...")
    run_command("kubectl apply -f manifests/crd-failurepredictor.yaml")
    
    # 2. Apply Sample CR
    print("Applying Sample FailurePredictor...")
    run_command("kubectl apply -f manifests/sample-failurepredictor.yaml")
    
    # 3. Start operator in background
    print("Starting Kopf Operator...")
    operator_proc = run_command("python -m kopf run src/operator.py --verbose", wait=False)
    
    print("Waiting 20 seconds for operator to run a few ticks...")
    time.sleep(20)
    
    # 4. Check CR Status
    print("Checking CR Status...")
    status_out = run_command("kubectl get failurepredictor train-ticket-predictor -o yaml")
    print("\n--- CR YAML ---")
    print(status_out)
    print("---------------\n")
    
    if "status:" in status_out and "risk:" in status_out:
        print("SUCCESS: Operator is running and updating the CRD status!")
    else:
        print("FAILURE: Status not found in CRD.")
        
    print("Stopping operator...")
    operator_proc.terminate()
    try:
        operator_proc.wait(timeout=5)
    except:
        operator_proc.kill()
        
if __name__ == "__main__":
    main()
