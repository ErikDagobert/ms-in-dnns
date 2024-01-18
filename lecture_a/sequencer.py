from math import sqrt
import argparse

parser = argparse.ArgumentParser(
                    prog='sequencer',
                    description='Produces sequences of a given length',
                    epilog=None)
parser.add_argument('--length', default='10', type=int)
parser.add_argument('--sequence', default="primes", type=str,
                    choices=["fibonacci", "prime", "square", "triangular", "factorial"],
                    help='Sequence choice')

args = parser.parse_args()


# Finds the next prime given a list of all previous primes
def nextPrime(primes):
    n = primes[-1]
    while (not isRelPrime(n, primes)):
        n = n + 1
    return n


# Checks if n is relatively prime with all numbers in a list
def isRelPrime(n, primes):
    val = True
    for p in primes:
        val = val and (n % p != 0)
    return val


def main(args):
    seq = []
    if args.sequence == "fibonacci":
        if args.length >= 1:
            seq.append(0)
        if args.length >= 2:
            seq.append(1)
        for n in range(2, args.length):
            seq.append(seq[-1] + seq[-2])
    elif args.sequence == "prime":
        if args.length >= 1:
            seq.append(2)
        for n in range(1, args.length):
            seq.append(nextPrime(seq))
    elif args.sequence == "square":
        if args.length >= 1:
            seq.append(1)
        for n in range(1, args.length):
            seq.append(round(seq[-1] + 2 * sqrt(seq[-1]) + 1))
    elif args.sequence == "triangular":
        if args.length >= 1:
            seq.append(1)
        for n in range(2, args.length + 1):
            seq.append(n + seq[-1])
    elif args.sequence == "factorial":
        if args.length >= 1:
            seq.append(1)
        for n in range(2, args.length + 1):
            seq.append(n * seq[-1])
    return seq


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
