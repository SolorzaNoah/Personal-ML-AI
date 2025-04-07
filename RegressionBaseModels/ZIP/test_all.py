import subprocess

scripts = [
    "train_regression1.py",
    "train_regression2.py",
    "train_regression3.py",
    "train_regression4.py",
    "train_regression_multi_output.py",
    "eval_regression1.py",
    "eval_regression2.py",
    "eval_regression3.py",
    "eval_regression4.py",
    "eval_regression_multi_output.py",
    "train_classifier1.py",
    "train_classifier2.py",
    "train_classifier3.py",
    "eval_classifier1.py",
    "eval_classifier2.py",
    "eval_classifier3.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script])
    print(f"Finished {script}.\n")
