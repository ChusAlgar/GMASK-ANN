from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import random



def kMedoids2(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}

    for t in range(tmax):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)

        # check for convergence
        if np.array_equal(M, Mnew):
            break

        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

def nGrams(text, n):
     return map(''.join, zip([text[i:] for i in range(n)]))

def nGramDistMatrix(X, n):
     m = len(X)
     D = np.zeros((m,m))
     for i in range(m):
          ngi = set(nGrams(X[i], n))
          lngi = len(ngi)
          for j in range(i+1,m):
              ngj = set(nGrams(X[j], n))
              lngj = len(ngj)
              lintersect = len(set.intersection(ngi,ngj))
              d = 1. - 2.
              lintersect / (lngi + lngj)
              D[i,j] = d
              D[j,i] = d
     return D


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

'''
X = np.array(['abe simpson','apu nahasapeemapetilon','barney gumbel','bart simpson','carl carlson',
             'charles montgomery burns','clancy wiggum','comic book guy','disco stu','dr. julius hibbert',
             'dr. nick riveria','edna krabappel','fat tony','gary chalmers','groundskeeper willie',
             'hans moleman','homer simpson','kent brockman','krusty the clown','lenny leonard','lisa simpson',
             'maggie simpson','marge simpson','martin prince','mayor quimby','milhouse van houten','moe syslak',
             'ned flanders','nelson muntz','otto mann','patty bouvier','prof. john frink','ralph wiggum',
             'reverend lovejoy','rod flanders','selma bouvier','seymour skinner','sideshow bob','snake jailbird',
             'todd flanders','waylon smithers'])

M, C = kMedoids(nGramDistMatrix(X, n=2), k=3)
print(X[M])
for c in range(3):
    print(X[C[c]])
'''


# 3 points in dataset
data = np.array([[1,1],
                [2,2],
                [10,10]])
data2 = np.random.randint(5, size=2560)
data2 = data2.reshape(10,256)

# distance matrix
D = pairwise_distances(data, metric='euclidean')

# split into 2 clusters
M, C = kMedoids(D, 2)

print('medoids:')
for point_idx in M:
    print( data[point_idx] )

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data2[point_idx]))
