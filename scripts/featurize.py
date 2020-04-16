from nff.data import Dataset

def main():
    pth = "dataset.pth.tar"
    dataset = Dataset.from_file(pth)
    dataset = Dataset(props=dataset.props, units=dataset.units)
    dataset.featurize()
    dataset.save(pth)

if __name__ == "__main__":
    main()