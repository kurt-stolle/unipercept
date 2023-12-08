"""
Run a command using the distributed engine.
"""


def main():
    from accelerate.commands.launch import main as launch_main

    launch_main()
