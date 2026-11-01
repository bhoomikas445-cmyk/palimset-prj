"""
PALIMPSEST Main Unified Entry Point
Run this script directly via:
    python main.py          (interactive menu or defaults to UI)
    python main.py --ui     (launches Streamlit UI)
    python main.py --api    (launches FastAPI server)
    python main.py --demo   (runs Drift-Recovery scenario)
    python main.py --eval   (runs 5-session benchmark)
    python main.py --test   (runs all pytest unit tests)
"""

import sys
import os
import argparse
import subprocess

def launch_ui():
    print("\n[PALIMPSEST] Starting Streamlit UI on http://localhost:8501 ...")
    app_path = os.path.join(os.path.dirname(__file__), "ui", "streamlit_app.py")
    try:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", app_path]
        sys.exit(stcli.main())
    except Exception:
        # Fallback to subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

def launch_api():
    print("\n[PALIMPSEST] Starting FastAPI server on http://127.0.0.1:8000 ...")
    import uvicorn
    from palimpsest.api.app import app
    uvicorn.run(app, host="127.0.0.1", port=8000)

def run_demo():
    from demo.demo_drift_recovery import run_drift_recovery_demo
    run_drift_recovery_demo()

def run_evaluation():
    from evaluation.run_eval import run_evaluation as execute_eval
    execute_eval()

def run_tests():
    import pytest
    print("\n[PALIMPSEST] Running pytest test suite...")
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    exit_code = pytest.main(["-v", test_dir])
    sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(description="PALIMPSEST Context Memory System")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI")
    parser.add_argument("--api", action="store_true", help="Start FastAPI REST Server")
    parser.add_argument("--demo", action="store_true", help="Run Drift-Recovery Demo")
    parser.add_argument("--eval", action="store_true", help="Run 5-Session Evaluation Benchmark")
    parser.add_argument("--test", action="store_true", help="Run Unit Test Suite")
    
    args = parser.parse_args()

    if args.ui:
        launch_ui()
    elif args.api:
        launch_api()
    elif args.demo:
        run_demo()
    elif args.eval:
        run_evaluation()
    elif args.test:
        run_tests()
    else:
        # Interactive selection
        print("=" * 60)
        print("           PALIMPSEST CONTEXT MEMORY SYSTEM           ")
        print("=" * 60)
        print("Select an option to run:")
        print("  [1] Launch Streamlit Interactive UI (Recommended)")
        print("  [2] Run Drift-Recovery Demo")
        print("  [3] Run 5-Session Evaluation Benchmark")
        print("  [4] Start FastAPI Server")
        print("  [5] Run Unit Tests")
        print("=" * 60)

        # Check if running interactively
        if sys.stdin.isatty():
            try:
                choice = input("Enter choice [1-5] (default: 1): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = "1"
        else:
            # Non-interactive default to UI or demo
            choice = "1"

        if choice == "2":
            run_demo()
        elif choice == "3":
            run_evaluation()
        elif choice == "4":
            launch_api()
        elif choice == "5":
            run_tests()
        else:
            launch_ui()

if __name__ == "__main__":
    main()
