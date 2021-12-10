# CS-paper
# Data libraries
import json
from joblib import load, dump

# Computation libraries
import numpy as np
import random

# Utility libraries
import copy
import re
from matplotlib import pyplot as plt

# Print options
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def import_data(path='Data/TVs-all-merged.json'):
    """Method to import data"""
    file = open(path, )
    data = json.load(file)
    file.close()
    return data


def feature_count(data):
    """Method to find most occurring features"""
    featureCount = {}
    for screen, info in data.items():
        for feature in info[0]["featuresMap"]:
            if feature in featureCount:
                featureCount[feature] = featureCount[feature] + 1
            else:
                featureCount[feature] = 1
    featureCountSorted = dict(reversed(sorted(featureCount.items(), key=lambda item: item[1])))
    for feature, count in featureCountSorted.items():
        print(f"{feature}: {count}")
    return featureCountSorted


def get_titles():
    """Method to retrieve titles from the data"""
    data = import_data()
    titles = []
    for screen, info in data.items():
        for dup in info:
            titles.append(dup['title'])
    return titles


def get_model_words():
    """Method to retrieve model words from the titles"""
    data = import_data()
    model_words = []
    for screen, info in data.items():
        for dup in info:
            all_model_words = re.finditer(r"[a-zA-Z0-9]*(([0-9]+[ˆ0-9ˆ,ˆ ]+)|([ˆ0-9ˆ,ˆ ]+[0-9]+))[a-zA-Z0-9]*", dup['title'])
            mws = []
            for mw in all_model_words:
                mws.append(mw.group())
            model_words.append(mws)
    return model_words


def kshingles(doc, k):
    """Method to compute k-shingles for a document"""
    shingles = []
    for i in range(len(doc) - k + 1):
        shingle = ''
        for j in range(k):
            shingle += doc[i + j]
        shingles.append(shingle)
    return shingles


def all_kshingles(docs, k):
    """Method to find all shingles from several documents"""
    shingles = []
    for doc in docs:
        shingles.append(kshingles(doc, k))
    return shingles


def boolean_matrix(title_mws, individual_words=False):
    """Method to compute the boolean matrix (one-hot encoding) for the title words (can be split)"""
    # Make universal set
    try:
        universal_set = set(title_mws)
    except Exception as e:
        universal_set = set([item for sublist in title_mws for item in sublist])

        # This is the part where my programming buddy Sara Stip (480355) and I deviate in our methodology.
        # Given the model words, I split these into parts, e.g. 'LED 720p' will become ['LED' and '720p'].
        # Sara will keep the model words as they are, so the example will remain 'LED 720p'.
        if individual_words:
            universal_list0 = []
            for mw in universal_set:
                for word in mw.split():
                    universal_list0.append(word)
            universal_set = set(universal_list0)
    universal_list = list(universal_set)

    # Compute boolean matrix
    b_matrix = np.ndarray(shape=(len(universal_list), len(title_mws)))
    for i in range(len(universal_list)):
        for j in range(len(title_mws)):
            if individual_words:
                mws = []
                for k in title_mws[j]:
                    for s in k.split():
                        mws.append(s)
                b_matrix[i, j] = 1 if universal_list[i] in mws else 0
            else:
                b_matrix[i, j] = 1 if universal_list[i] in title_mws[j] else 0
    return b_matrix


def prime_list(ub):
    """Very fast method to compute primes up to an upper bound"""
    out = list()
    sieve = [True] * (ub + 1)
    for p in range(2, ub + 1):
        if sieve[p]:
            out.append(p)
            for i in range(p, ub + 1, p):
                sieve[i] = False
    return out


def signature_matrix(bm, p=100):
    """Method to compute signature matrix for p permutations"""
    sm = np.ndarray(shape=(p, bm.shape[1]))
    perm = list(range(1, bm.shape[0] + 1))
    for i in range(p):
        random.shuffle(perm)
        for c in range(bm.shape[1]):
            ones = list(np.nonzero(bm[:, c])[0])
            perm_ones = np.array(perm)[ones]
            sm[i, c] = min(perm_ones)
    return sm


def LSH(sm, t):
    """Method to gather candidate pairs through the LSH algorithm"""
    # Initialization
    n = sm.shape[0]
    divisors = []
    for i in range(1, n+1):
        if n % i == 0:
            divisors.append(i)
    tees = []
    for bees in divisors:
        base = 1/bees
        power = 1/(n/bees)
        tees.append(base**power)
    index = np.argmin(np.abs(np.array(tees) - t))
    b = divisors[index]
    # print(f'Bands b: {b}, Rows r: {n/b}')

    sm_split = np.array_split(sm, b)
    candidate_pairs = []
    for band in sm_split:
        band_cols = []
        for col in range(band.shape[1]):
            # Compute hash function for column 'col' in a certain band
            mod = list(band[:, col])
            band_cols.append(mod)
        for j1 in range(len(band_cols)):
            for j2 in range(j1+1, len(band_cols)):
                if band_cols[j1] == band_cols[j2]:
                    candidate_pairs.append([j1, j2])

    # Compute unique pairs
    unique_candidate_pairs = [list(up) for up in set(tuple(up) for up in candidate_pairs)]
    return unique_candidate_pairs


def jaccard_similarity(col1, col2):
    """Method to calculate Jaccard similarity between 2 columns"""
    numerator = 0
    denominator = 0
    for i in range(len(col1)):
        if col1[i] == 1 and col2[i] == 1:
            numerator += 1
            denominator += 1
        elif col1[i] == 1 or col2[i] == 1:
            denominator += 1
    return numerator/denominator


def similarity_for_pairs(candidate_pairs, bm):
    """Calculate Jaccard similarity for all candidate pairs"""
    j_sim = []
    for pair in candidate_pairs:
        j_sim.append(jaccard_similarity(bm[:, pair[0]].tolist(), bm[:, pair[1]].tolist()))
    return j_sim


def get_duplicates():
    """Find duplicates through modelID in data (this is usually not possible, only used for evaluation)"""
    data = import_data()
    model_words = get_model_words()
    duplicate_matrix = np.zeros(shape=(len(model_words), len(model_words)))
    duplicates = {}
    seen = []
    duplicate_pairs = []
    for screen, info in data.items():
        for dup in info:
            if dup['modelID'] not in seen:
                seen.append(dup['modelID'])
                duplicates[dup['modelID']] = 1
            elif dup['modelID'] in seen:
                for i in np.where(np.array(seen) == dup['modelID'])[0].tolist():
                    duplicate_pairs.append([i, len(seen)])
                seen.append(dup['modelID'])
                duplicates[dup['modelID']] += 1

    # Trim duplicates and count amount of duplicates
    duplicates_copy = copy.copy(duplicates)
    for key, val in duplicates_copy.items():
        if val < 2:
            del duplicates[key]
    duplicates_count = len(duplicate_pairs)

    # Fill duplicate matrix
    for pair in duplicate_pairs:
        duplicate_matrix[pair[0], pair[1]] = 1

    return duplicate_matrix, duplicates, duplicates_count


def percentile_pairs(bm, candidate_pairs, p):
    """Find pairs which Jaccard similarity is greater than or equal to p"""
    j_sim = similarity_for_pairs(candidate_pairs, bm)
    pairs = dict(zip(j_sim, candidate_pairs))
    pairs_p = []
    for key, val in pairs.items():
        if key >= p:
            pairs_p.append(val)
    return pairs_p


def main():
    """Main method for running code"""
    # Find duplicates
    amount_of_pairs = 1317876
    duplicate_matrix, duplicates, duplicates_count = get_duplicates()
    print(f'Amount of duplicates: {duplicates_count}')
    print(f'')

    # Compute/load boolean matrix and signature matrix
    bm = load_file(name='bm', verbose=False)
    sm = load_file(name='sm', verbose=False)
    bm_iw = load_file(name='bm_iw', verbose=False)
    sm_iw = load_file(name='sm_iw', verbose=False)

    # Get candidate pairs from LSH algorithm
    t_vals = [x / 10 for x in list(range(1, 11))]
    for t in t_vals:
        print(f't = {t}')
        candidate_pairs = LSH(sm, t=t)
        print(f'Candidate pairs: {len(candidate_pairs)}')

        # Find correct candidate pairs
        correct_candidate_pairs = 0
        for pair in candidate_pairs:
            if duplicate_matrix[pair[0], pair[1]] == 1:
                correct_candidate_pairs += 1
        print(f'Correct candidate pairs: {correct_candidate_pairs}')

        # Evaluation metrics
        pq = correct_candidate_pairs/len(candidate_pairs)
        pc = correct_candidate_pairs/duplicates_count
        F1 = 2*pq*pc / (pq + pc)
        frac_of_comp = len(candidate_pairs)/amount_of_pairs
        # print(f'')
        print(f'Pair Quality: {pq}')
        print(f'Pair Completeness: {pc}')
        print(f'F1-measure: {F1}')
        print(f'Fraction of comparisons: {frac_of_comp}')
        print(f'')


    t_vals = [x / 10 for x in list(range(1, 11))]
    pq, pc, F1, frac_of_comp = graph(t_vals)

    plot1 = plt.figure(1)
    plt.plot(t_vals, pq)
    plt.xlabel('t')
    plt.ylabel('Pair quality')

    plot2 = plt.figure(2)
    plt.plot(t_vals, pc)
    plt.xlabel('t')
    plt.ylabel('Pair completeness')

    plot3 = plt.figure(3)
    plt.plot(t_vals, F1)
    plt.xlabel('t')
    plt.ylabel('F1-score')

    plot4 = plt.figure(4)
    plt.plot(frac_of_comp, pq)
    plt.xlabel('Fraction of comparisons')
    plt.ylabel('Pair quality')

    plot5 = plt.figure(5)
    plt.plot(frac_of_comp, pc)
    plt.xlabel('Fraction of comparisons')
    plt.ylabel('Pair completeness')

    plot6 = plt.figure(6)
    plt.plot(frac_of_comp, F1)
    plt.xlabel('Fraction of comparisons')
    plt.ylabel('F1-score')

    plt.show()


def graph(t_vals):
    amount_of_pairs = 1317876
    duplicate_matrix, duplicates, duplicates_count = get_duplicates()
    sm = load_file(name='sm_iw', verbose=False)
    y_pq = []
    y_pc = []
    y_F1 = []
    y_frac_of_comp = []
    for t in t_vals:
        candidate_pairs = LSH(sm, t=t)

        # Find correct candidate pairs
        correct_candidate_pairs = 0
        for pair in candidate_pairs:
            if duplicate_matrix[pair[0], pair[1]] == 1:
                correct_candidate_pairs += 1

        # Evaluation metrics
        pq = correct_candidate_pairs / len(candidate_pairs)
        pc = correct_candidate_pairs / duplicates_count
        F1 = 2 * pq * pc / (pq + pc)
        frac_of_comp = len(candidate_pairs) / amount_of_pairs
        y_pq.append(pq)
        y_pc.append(pc)
        y_F1.append(F1)
        y_frac_of_comp.append(frac_of_comp)
    return y_pq, y_pc, y_F1, y_frac_of_comp


def save_file(file, name):
    """Method to save a compressed boolean matrix"""
    try:
        dump(file, f'{name}.joblib', compress=True)
        print(f"Saving succeeded!")
    except Exception as e:
        raise IOError(f"Error saving boolean matrix: {str(e)}")


def load_file(name, verbose=True):
    """Method to load boolean matrix"""
    try:
        bm = load(f"{name}.joblib")
        if verbose:
            print(f"Loading succeeded!")
        return bm
    except FileNotFoundError:
        print(f"File not found.")


# Used in case several Python files need to be used in tandem
if __name__ == '__main__':
    main()

