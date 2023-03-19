import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hi', default=None, type=str)
    args = parser.parse_args()
    print(args)
    hi = args.hi
    print(hi==None)
    print(hi == "None")
