import numpy as np
import math


def sample_from_unit_ball(m, n, mode='uniform'):
    # assertions on input
    assert type(m) == int and type(n) == int
    assert m > 0 and n > 0
    # sample in n-dimensional space, each coordinate being normally distributed
    # (this distribution is isotropic!)
    samples = np.random.randn(m, n)
    # for sampling on the surface, scale each point to norm 1
    if mode == 'uniform' or mode == 'surface':
        samples = samples / np.linalg.norm(samples, axis=1).reshape((-1, 1))
    # when sampling uniformly in the ball, give the points on the surface
    # an appropriately sampled radius
    if mode == 'uniform':
        # todo!!!
        pass
    # otherwise (e.g., mode='normal') keep the normal samples and return
    return samples


def clustering_multiple_k(data, min_k=1, max_k=9,
                          max_iter=50, min_ratio_mk=2):
    # assertions on input and preparations
    assert all([type(min_k) == int, type(min_k) == int, min_k > 0])
    assert type(data) == np.ndarray and len(data.shape) == 2
    m = data.shape[0]
    max_k = min(max_k, int(m/min_ratio_mk))
    # initialise dictionary of classes and volumes
    class_dict = {}
    vol_dict = {}
    # iterate over k in the given range, collect class labels and the
    # 'volume' of the clustering in each case (the keys of those dictionaries
    # are 'k=1', 'k=2', ... to avoid confusion)
    for k_iter in range(min_k, max_k+1):
        key = 'k='+str(k_iter)
        class_dict[key], vol_dict[key] = clustering_fixed_k(data, k_iter,
                                                            max_iter,
                                                            min_ratio_mk)
    return class_dict, vol_dict


def clustering_fixed_k(data, k, max_iter=50, min_ratio_mk=2):
    # preparations and assertions
    assert type(k) == int and k > 0
    assert type(data) == np.ndarray and len(data.shape) == 2
    m, n = data.shape
    assert m >= 2*k
    # set volume of n-dimensional unit ball
    # (cf. https://en.wikipedia.org/wiki/Volume_of_an_n-ball)
    Vn = math.pi**int(n/2) / math.gamma(n/2+1)
    # intialise classes
    classes = np.array(list(range(k))*int(np.ceil(m/k)))[:m]
    np.random.shuffle(classes)
    # intialise centres and make bigger matrix of data for vectorised
    # operations below
    temp_centres = np.zeros((m, n, k))
    temp_data = np.broadcast_to(data.reshape(m, n, 1), (m, n, k))
    # clustering algorithm
    for i in range(max_iter):
        # make copy for comparison later
        old_classes = classes
        # find class centres
        for k_iter in range(k):
            temp_centres[:, :, k_iter] = np.broadcast_to(
                (np.sum(data[classes == k_iter, :], axis=0)
                 / sum(classes == k_iter)).reshape((1, n)), (m, n))
        # now compute distances of samples to class centres:
        # the entry (m,k) of the vector distances in the following line
        # is the distance between the sample m and class centre k
        distances = np.sqrt(np.sum(np.power(temp_data-temp_centres, 2),
                                   axis=1))
        # set class of each sample to the centre it is closest to
        classes = np.argmin(distances, axis=1)
        # if a class has dropped out, print a warning and stop
        if len(np.unique(classes)) < k:
            print('k=' + str(k) + ': Warning: class dropped out')
            return None, None
        # check condition for termination of iteration
        if (all(old_classes == classes)) or (i == max_iter - 1):
            # determine the sum of the volume of the balls
            volume = 0
            for j in range(k):
                volume += Vn * (distances[classes == j, j].max())**n
            # message to state how many iterations were used
            # or give a warning
            if (i == max_iter - 1):
                print('k=' + str(k)
                      + ': Warning: reached maximum number of iterations')
            else:
                print('k=' + str(k)
                      + ': Stopped clustering after iteration ' + str(i))
            break
    return classes, volume
