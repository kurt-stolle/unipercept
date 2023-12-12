"""
Run a command using the distributed engine.

We use HuggingFace Accelerate to do the heavy lifting.
"""


def main():
    from accelerate.commands.launch import main as launch_main

    launch_main()
