from nff.utils.confgen import confs_and_save
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        help="Path to JSON file with confgen information")
    args = parser.parse_args()
    confs_and_save(config_path=args.config_path)


if __name__ == "__main__":
    main()
