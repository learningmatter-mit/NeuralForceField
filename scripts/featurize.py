from nff.data import Dataset
import sys
sys.path.insert(0, "..")


def main():
    pth = "dataset.pth.tar"
    dataset = Dataset.from_file(pth)
    dataset = Dataset(props=dataset.props, units=dataset.units)
    dataset.featurize()
    dataset.save(pth)


if __name__ == "__main__":
    main()
