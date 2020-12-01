import os

BASE = "/home/saxelrod/chemprop_cov_1"
NAMES = ['train_full_old', 'val_full_old', 'test_full_old']
PATHS = [os.path.join(BASE, name + ".csv") for name in NAMES]

def main(paths=PATHS):
    for path in paths:
        new_path = path.replace("_old", "")
        with open(path, 'r') as f:
            lines = f.readlines()
        dic = {i: line for i, line in enumerate(lines) if len(line.split(",")) == 3}
        new_lines = []
        for i, line in enumerate(lines):
            if i in dic:
                new_split = line.split(",")
                middle = new_split[1]
                start_digit = middle[0]
                middle = [f"{start_digit}\n", middle.replace(start_digit, "")]
                new_split = [new_split[0], *middle, *new_split[2:]]
                new_lines += [",".join(new_split[:2]), ",".join(new_split[2:])]
            else:
                new_lines.append(line)

        new_text = "".join(new_lines)
        with open(new_path, "w") as f:
            f.write(new_text)

if __name__ == "__main__":
    main()
