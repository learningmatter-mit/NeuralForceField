import sys
sys.path.insert(0, "$HOME/repo/nff/covid/NeuralForceField")

from nff.data import Dataset

def main():
    pth = "dataset.pth.tar"
    dataset = Dataset.from_file(pth)
    dataset = Dataset(props=dataset.props, units=dataset.units)
    dataset.featurize()
    dataset.save(pth)

if __name__ == "__main__":
    main()