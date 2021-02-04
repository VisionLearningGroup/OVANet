import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm


def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape

    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weight(source_feature, target_feature, validation_feature):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    source_feature = source_feature.copy()
    target_feature = target_feature.copy()
    rand_s = np.random.permutation(source_feature.shape[0])
    rand_t = np.random.permutation(target_feature.shape[0])

    source_feature = source_feature[rand_s[:3000]]
    target_feature = target_feature[rand_t[:3000]]
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)
    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)

    decays = np.logspace(-2, 4, 5)#[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    val_acc = []
    domain_classifiers = []

    for decay in decays:
        domain_classifier = svm.SVC(C=decay, kernel='linear', verbose=False, probability=True, max_iter=4000)    #MLPClassifier(hidden_layer_sizes=(d, 256, 2), activation='relu', alpha=decay)

        #domain_classifier = svm.SVC(C=decay, kernel='linear', verbose=False, probability=True)    #MLPClassifier(hidden_layer_sizes=(d, 256, 2), activation='relu', alpha=decay)

        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        domain_out = domain_classifier.predict_proba(target_feature)
        deceive_score = domain_out[:, 1:]
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)
        print('decay is %s, val acc is %s deceive score %s' % (decay, acc, np.mean(deceive_score)))

    index = val_acc.index(max(val_acc))

    print('val acc is')
    print(val_acc)

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t