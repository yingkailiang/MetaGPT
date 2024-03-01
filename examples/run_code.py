import subprocess

# Run a command and check its exit code
try:
  output = subprocess.run(["python3", "test.py"], check=True)
  print(output.stdout)
except subprocess.CalledProcessError as e:
  print("Error:", e)