import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10_000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: dict[str, set[str]], page: str, damping_factor: float):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # P(page = i) = P(page = i, damping)             + P(page = i, !damping) 
    #             = P(page = i | damping)*P(damping) + P(page = i | !damping)*P(!damping)
    num_pages = len(corpus)
    assert num_pages > 0

    num_links = len(corpus[page])
    if num_links == 0:
        return {p: 1/num_pages for p in corpus.keys()}

    probs = {page: (1-damping_factor) / num_pages for page in corpus.keys()}
    for link in corpus[page]:
        probs[link] += damping_factor / num_links
    return probs


def sample_pagerank(corpus: dict[str, set[str]], damping_factor: float, n: int):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    assert n > 1

    sample = {page: 0 for page in corpus}
    cur = random.choice(list(corpus.keys()))
    while n > 0:
        sample[cur] += 1
        probs = transition_model(corpus, cur, damping_factor)
        cur = random.choices(list(probs.keys()), probs.values())[0]
        n -= 1
    total = sum(sample.values())
    return {page: count/total for page, count in sample.items()}


def iterate_pagerank(corpus: dict[str, set[str]], damping_factor: float):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    epsilon = 0.001
    num_pages = len(corpus)
    pr = {page: 1/num_pages for page in corpus.keys()}

    def calculate(page: str):
        link_weight = 0
        for p in corpus:
            num_links = len(corpus[p])
            if num_links == 0:
                link_weight += pr[p] / num_pages
            elif page in corpus[p]:
                link_weight += pr[p] / num_links
        return (1-damping_factor)/num_pages + damping_factor * link_weight

    converged = False
    while not converged:
        converged = True
        for page in pr.keys():
            prev, pr[page] = pr[page], calculate(page)
            diff = abs(prev - pr[page])
            converged = diff < epsilon
    return pr


if __name__ == "__main__":
    main()
