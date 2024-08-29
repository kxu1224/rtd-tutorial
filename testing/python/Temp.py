# def PilotExperiment(dataname, pilot_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None):
#     r""" 
#         This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
#         with several pilot size given data, model, batch_size, learning_rate, epoch, off_aug and pre_model.
#         For each pilot size, there will be 5 draws.
#         For each draw, the data is augmented to 5 times the original sample size.

#         Parameters
#         ----------
#         dataname : string
#                    pure data name without .csv. Eg: SKCMPositive_3
#         pilot_size : list
#                      a list including potential pilot sizes
#         model : string
#                 name of the model to be trained
#         batch_frac : float
#                       batch fraction
#         learning_rate : float
#                 learning rate 
#         epoch : int
#                                 choose from None (early_stop), or any integer, if choose None, early_stop_num will take effect
#         early_stop_num : int
#               if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
#         off_aug : string (AE_head or Gaussian_head or None)
#                             choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
#         AE_head_num : int
#                     how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
#         Gaussian_head_num : int
#                 how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
#         pre_model : string
#                         transfer learning input model. If pre_model == None, no transfer learning 

#     """
#     print("Pilot Experiment Start")
#     return None

# #%% Define application of experiment
# def ApplyExperiment(path, dataname, apply_log, new_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None, save_model = None):
#     r"""
#         This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
#         given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
#         and generate new samples with size specified by the users.

#     Parameters
#     ----------
#     path : string
#                               path for reading real data and saving new data
#     dataname : string
#                     pure data name without .csv. Eg: SKCMPositive_3
#     apply_log : boolean
#                       logical whether apply log transformation before training
#     model : string
#                               name of the model to be trained
#     batch_frac : float
#                     batch fraction
#     learning_rate : float
#               learning rate 
#     epoch : int
#                               choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
#     early_stop_num : int
#             if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
#     off_aug : string (AE_head or Gaussian_head or None) 
#                       choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
#     AE_head_num : int
#                   how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
#     Gaussian_head_num : int
#          how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
#     pre_model : string
#                       transfer learning input model. If pre_model == None, no transfer learning
#     save_model : string
#                     if the trained model should be saved, specify the path and name of the saved model
#     """
    
#     print("Apply Experiment Start")
#     return None

# #%% Define transfer learing 
# def TransferExperiment(pilot_size, fromname, toname, fromsize, model, new_size=500, apply_log=True, epoch=None, batch_frac=0.1, learning_rate=0.0005, off_aug=None):
#     """
#         This function run transfer learning using VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
#         given fromdata, todata, model, batch_size, learning_rate, epoch, off_aug and pre_model
#         and generate new samples with size specified by the users.
#         The fine tuning model training can be pilot experiments or apply experiment
#         Make sure data files for pre_model training and fine tuning model training are in Transfer/

#     Parameters
#     ----------
#     pilot_size : int
#                     if None, the fine tuning model will be apply experiment and new_size will take effect
#                     otherwise, the fine tuning model will be trained using pilot experiments
#     fromname : string
#                         the dataname for pre_model training 
#     toname : string 
#                             the dataname for fine tuning model training
#     fromsize : int
#                         the sample size of the fromdata
#     new_size : int
#                         if apply experiment, this will be the sample size of generated samples
#     apply_log : boolean
#                       logical whether apply log transformation before training
#     model : string
#                               name of the model to be trained
#     batch_frac : float
#                     batch fraction
#     learning_rate : float
#               learning rate 
#     epoch : int
#                               choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
#     off_aug : string (AE_head or Gaussian_head or None)
#                           choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation"""

#     print("Transfer Learning Start")
#     return None


def LOGIS(train_data, train_labels, test_data, test_labels):
    r"""This is an L1 or Lasso regression classifier. The 'liblinear' solver is used because it is recommended for small datasets and L1 penalty.
    Cs represents the inverse of regularization strength and is set to 10; smaller values specify stronger regularization.
    
    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data

    """
    print("logis")
    return None

def SVM(train_data, train_labels, test_data, test_labels):
    r"""This is a Support Vector Machine classifier

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    print("svm")
    return None


def KNN(train_data, train_labels, test_data, test_labels):
    r"""This is a K-Nearest Neighbor classifier with k = 20

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    print("knn")
    return None

def RF(train_data, train_labels, test_data, test_labels):
    r"""This is a Random Forest classifier

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    print("rf")
    return None

def XGB(train_data, train_labels, test_data, test_labels):
    r"""This is an XGBoost classifier. 
    
    In the paper XGBoost is done with 25 rounds for miRNA data and 10 rounds for RNA data. For RNA-
    seq data, specific XGBoost parameters were adjusted: the learning rate was set to 0.1, the maximum depth
    of a tree was set to 3, and the minimum sum of instance weight (hessian) needed in a child was set to 3;
    other training parameters were kept at their default values.


    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    print("xgb")
    return None




def heatmap_eval(dat_generated, dat_real):
    r"""
    This function creates a heatmap visualization comparing the generated data and the real data

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the data generated from ApplyExperiment
    dat_real: pd.DataFrame
            the original copy of the data
    
    """
    print("heatmap")
    return None

def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, legend_pos="top"):
    r"""
    This function creates a UMAP visualization comparing the generated data and the real data

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the data generated from ApplyExperiment
    dat_real: pd.DataFrame
            the original copy of the data
    groups_generated : pd.Series
            the groups generated
    groups_real : pd.Series
            the real groups
    legend_pos : string
            legend location
    
    """
    print("umap")
    return None

def eval_classifier(whole_generated, whole_groups, n_candidate, n_draw=5, log=True):
    r"""
    This method assesses the classifiers’ performance based on classification accuracy computed through 5-fold cross-validation

    Parameters
    -----------
    whole_generated : pd.DataFrame
            the entire set of generated data
    whole_groups: pd.DataFrame
            all the available groups
    n_candidate : int
            the number of candidates
    n_draw : int, optional
            the number of cross-validations
    log : boolean, optional
            option to take log of the data


    """
    print("evals!")
    return None


def fit_curve(acc_table, metric_name, n_target=None, plot=True, ax=None, annotation=("Metric", "")):
    r"""
    
    
    """

    print("fit curve")
    return None


def vis_classifier(metric_real, n_target, metric_generated = None):
    r""" Method to visualize the classification results for both real and generated data.
    
    Parameters
    -----------
    metric_real : pd.DataFrame
            real dataset values
    n_target: int
            number of targets to fit the learning curve
    metric_generated : pd.DataFrame, optional
            generated dataset values
    
    
    """

    print("vis")
    return None