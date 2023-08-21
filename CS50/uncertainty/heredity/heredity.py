import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01,
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people: dict[str, dict], one_gene: set[str], two_genes: set[str], have_trait: set[str]):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    # Probability to inherit gene from parent with gene count
    INHERIT = {
        0: PROBS['mutation'],
        1: 0.5,
        2: (1 - PROBS['mutation']),
    }

    def copies(name: str):
        if name in one_gene: return 1
        elif name in two_genes: return 2
        else: return 0

    joint_prob = 1
    for name, data in people.items():
        genes = copies(name)
        if (data['mother'], data['father']) != (None, None):
            # Probabilities that the child inherits gene from mother, father
            m, f = INHERIT[copies(data['mother'])], INHERIT[copies(data['father'])]
            gene_p = {
                0: (1-m)*(1-f),
                1: f*(1-m) + m*(1-f),
                2: f*m,
            }
        else:
            gene_p = PROBS['gene']
        joint_prob *= gene_p[genes] * PROBS['trait'][genes][name in have_trait]
    return joint_prob


def update(probabilities: dict[str, dict], one_gene: set, two_genes: set, have_trait: set, p: float):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name, probs in probabilities.items():
        if name in one_gene:
            probs['gene'][1] += p
        elif name in two_genes:
            probs['gene'][2] += p
        else:
            probs['gene'][0] += p
        probs['trait'][name in have_trait] += p


def normalize(probabilities: dict[str, dict]):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities.values():
        s = sum(person['gene'].values())
        for k, v in person['gene'].items():
            person['gene'][k] = v / s

        s = sum(person['trait'].values())
        for k, v in person['trait'].items():
            person['trait'][k] = v / s
        


if __name__ == "__main__":
    main()
