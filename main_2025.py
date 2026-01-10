"""
for target tracking
"""
import pathlib



CONFIG_FILE_PATH = pathlib.Path("config.yaml")


def main() -> int:
    """Main Function for target tracking"""

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
