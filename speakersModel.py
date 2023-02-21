from functions import *

def train_model():
    featuresHabiba = extractFeaturesForFolder('./Habiba/')
    Habiba_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='diag',n_init =3)
    Habiba_gmm.fit(featuresHabiba)
    shirouq_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='diag',n_init =3)
    featuresShirouq = extractFeaturesForFolder('./Shirouq/')
    shirouq_gmm.fit(featuresShirouq)
    Rawda_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='diag',n_init =3)
    featuresRawda = extractFeaturesForFolder('./Rawda/')
    Rawda_gmm.fit(featuresRawda)
    pickle.dump(Habiba_gmm,open('trained_models/Habiba.gmm','wb'))
    pickle.dump(shirouq_gmm,open('trained_models/shirouq.gmm','wb'))
    pickle.dump(Rawda_gmm,open('trained_models/Rawda.gmm','wb'))

train_model()