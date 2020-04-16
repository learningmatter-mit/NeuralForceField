import sys
sys.path.insert(0, "..")

from nff.data import Dataset

def main():
    pth = "dataset.pth.tar"
    dataset = Dataset.from_file(pth)
    dataset.featurize()
    dataset.save(pth)

if __name__ == "__main__":
    main()