import sys
import os

print(f"{os.path.dirname(os.path.realpath(__file__))}/src")
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/src")
print(f"Current Python path: {sys.path}")
