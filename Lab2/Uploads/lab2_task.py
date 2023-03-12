import numpy as np

class NaiveBayesClassifier:
  def __init__(self):
    self.train_dataset = None
    self.train_labels = None
    self.train_size = 0
    self.num_features = 0
    self.num_classes = 0

  def fit(self, train_dataset, train_labels):
    self.train_dataset = train_dataset
    self.train_labels = train_labels
    # TODO
    self.train_size = train_dataset.shape[0]  #900
    self.num_features = train_dataset.shape[1]  #2642
    self.num_classes = np.amax(train_labels) + 1  #3
  
  def estimate_class_prior(self):
    # TODO
    # go through the array for calculating 0, 1, 2 probs
    class_prior = np.array([((self.train_labels==0).sum() + 1) / (self.train_size+self.num_classes), ((self.train_labels==1).sum() + 1) / (self.train_size+self.num_classes), ((self.train_labels==2).sum() + 1) / (self.train_size+self.num_classes)])
    return class_prior

  def estimate_likelihoods(self):
    # TODO
    likelihoods = np.zeros((2642, 3))
    train_labels_0 = (np.sum((self.train_dataset.T==True) * (self.train_labels==0), axis = 1) + 1) / ((self.train_labels==0).sum() + 2)
    train_labels_1 = (np.sum((self.train_dataset.T==True) * (self.train_labels==1), axis = 1) + 1) / ((self.train_labels==1).sum() + 2)
    train_labels_2 = (np.sum((self.train_dataset.T==True) * (self.train_labels==2), axis = 1) + 1) / ((self.train_labels==2).sum() + 2)
    likelihoods = np.stack((train_labels_0, train_labels_1, train_labels_2), axis=1)
    # sum_train_labels_0 = (self.train_labels==0).sum()
    # sum_train_labels_1 = (self.train_labels==1).sum()
    # sum_train_labels_2 = (self.train_labels==2).sum()
    # for i in range(2642):
    #   likelihoods[i][0] = (np.logical_and((self.train_labels==0),(self.train_dataset.T[i]==True)).sum() + 1) / (sum_train_labels_0 + 2)
    # for i in range(2642):
    #   likelihoods[i][1] = (np.logical_and((self.train_labels==1),(self.train_dataset.T[i]==True)).sum() + 1) / (sum_train_labels_1 + 2)
    # for i in range(2642):
    #   likelihoods[i][2] = (np.logical_and((self.train_labels==2),(self.train_dataset.T[i]==True)).sum() + 1) / (sum_train_labels_2 + 2)
    return likelihoods

  def predict(self, test_dataset):
    class_prior = self.estimate_class_prior()
    yes_likelihoods = self.estimate_likelihoods()
    no_likelihoods = 1 - yes_likelihoods
    # TODO
    result_cmp = np.log(class_prior) + (test_dataset==True) @ np.log(yes_likelihoods) + (test_dataset==False) @ np.log(no_likelihoods)
    test_predict = np.argmax(result_cmp, axis = 1)
    # for i in range(100):
    #   result_cmp[i][0] += np.log(class_prior[0])
    #   result_cmp[i][1] += np.log(class_prior[1])
    #   result_cmp[i][2] += np.log(class_prior[2])
    #   for j in range(2642):
    #     if(test_dataset[i][j] == True):
    #       result_cmp[i][0] += np.log(yes_likelihoods[j][0])
    #       result_cmp[i][1] += np.log(yes_likelihoods[j][1])
    #       result_cmp[i][2] += np.log(yes_likelihoods[j][2])
    #     elif(test_dataset[i][j] == False):
    #       result_cmp[i][0] += np.log(no_likelihoods[j][0])
    #       result_cmp[i][1] += np.log(no_likelihoods[j][1])
    #       result_cmp[i][2] += np.log(no_likelihoods[j][2])
    # test_predict = np.argmax(result_cmp, axis = 1)
    return test_predict