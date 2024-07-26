import numpy as np
from itertools import product
import requests

def get_groupIndex_to_featureIndices(featureIndex_to_groupIndex):
    groupIndex_to_featureIndices = {}
    for featureIndex, groupIndex in enumerate(featureIndex_to_groupIndex):
        if groupIndex not in groupIndex_to_featureIndices:
            groupIndex_to_featureIndices[groupIndex] = set()
        groupIndex_to_featureIndices[groupIndex].add(featureIndex)
    return groupIndex_to_featureIndices

def get_support_indices(betas):
    return np.where(np.abs(betas) > 1e-9)[0]

def get_nonsupport_indices(betas):
    return np.where(np.abs(betas) <= 1e-9)[0]

def normalize_X(X):
    X_mean = np.mean(X, axis=0)
    X_norm = np.linalg.norm(X-X_mean, axis=0)
    scaled_feature_indices = np.where(X_norm >= 1e-9)[0]
    X_normalized = X-X_mean
    X_normalized[:, scaled_feature_indices] = X_normalized[:, scaled_feature_indices]/X_norm[[scaled_feature_indices]]
    return X_normalized, X_mean, X_norm, scaled_feature_indices

def compute_logisticLoss_from_yXB(yXB):
    # shape of yXB is (n, )
    return np.sum(np.log(1.+np.exp(-yXB)))

def compute_logisticLoss_from_ExpyXB(ExpyXB):
    # shape of ExpyXB is (n, )
    return np.sum(np.log(1.+np.reciprocal(ExpyXB)))

def compute_logisticLoss_from_betas_and_yX(betas, yX):
    # shape of betas is (p, )
    # shape of yX is (n, p)
    yXB = yX.dot(betas)
    return compute_logisticLoss_from_yXB(yXB)

def compute_logisticLoss_from_X_y_beta0_betas(X, y, beta0, betas):
    XB = X.dot(betas) + beta0
    yXB = y * XB
    return compute_logisticLoss_from_yXB(yXB)

def convert_y_to_neg_and_pos_1(y):
    y_max, y_min = np.min(y), np.max(y)
    y_transformed = -1 + 2 * (y-y_min)/(y_max-y_min) # convert y to -1 and 1
    return y_transformed

def isEqual_upTo_8decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-8
    return np.max(np.abs(a - b)) < 1e-8

def isEqual_upTo_16decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-16
    return np.max(np.abs(a - b)) < 1e-16

def insertIntercept_asFirstColOf_X(X):
    n = len(X)
    intercept = np.ones((n, 1))
    X_with_intercept = np.hstack((intercept, X))
    return X_with_intercept

def get_all_product_booleans(sparsity=5):
    # build list of lists:
    all_lists = []
    for i in range(sparsity):
        all_lists.append([0, 1])
    all_products = list(product(*all_lists))
    all_products = [list(elem) for elem in all_products]
    return np.array(all_products)

def download_file_from_google_drive(id, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # link: https://stackoverflow.com/a/39225272/5040208
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def check_bounds(bound,  bound_name, num_features):
    if isinstance(bound, (float, int)):
        assert bound >= 0 if bound_name == "ub" else bound <= 0, f"{bound_name} needs to be >= 0" if bound_name == "ub" else f"{bound_name} needs to be <= 0"
    elif isinstance(bound, list):
        bound = np.asarray(bound)
        assert len(bound) == num_features, f"{bound_name}s for the features need to have the same length as the number of features"
        assert np.all(bound >= 0 if bound_name == "ub" else bound <= 0), f"all of {bound_name}s needs to be >= 0" if bound_name == "ub" else f"all of {bound_name}s needs to be <= 0"
    else:
        raise ValueError(f"{bound_name} needs to be a float, int, or list")
    
    return bound

def get_continuousFeatureName_from_binaryFeatureName(binaryFeatureName):
    # convert binary feature name to continuous feature name
    # check whether binaryFeatureName is in the format 'FeatureName<=Threshold' or 'Threshold1<FeatureName<=Threshold2'
    featureName = None
    errorMessage = f"Feature name {binaryFeatureName} does not follow the format 'FeatureName<=Threshold' or 'Threshold1<FeatureName<=Threshold2' or FeatureName_isNaN!"

    if '_isNaN' in binaryFeatureName:
        num_isnan = binaryFeatureName.count('_isNaN')
        if num_isnan > 1:
            raise ValueError(f"{errorMessage}")
        if binaryFeatureName[-6:] != '_isNaN':
            raise ValueError(f"{errorMessage}")
        featureName = binaryFeatureName.split('_')[0]

    elif '<=' in binaryFeatureName:
        num_leq = binaryFeatureName.count('<=')
        num_less = binaryFeatureName.count('<')

        if num_less == 2 and num_leq == 1:
            # this is the case where the feature name is in the form of 'Threshold1<FeatureName<=Threshold2'
            try:
                featureName = binaryFeatureName.split('<')[1].split('<=')[0]
            except:
                raise ValueError(f"{errorMessage}")
            
        elif num_less == 1 and num_leq == 1:
            # this is the case where the feature name is in the form of 'FeatureName<=Threshold'
            try:
                featureName = binaryFeatureName.split('<=')[0]
            except:
                raise ValueError(f"{errorMessage}")
            
        else:
            raise ValueError(f"{errorMessage}")

    else:
        raise ValueError(f"{errorMessage}")

    if not isinstance(featureName, str):
        raise ValueError(f"{errorMessage}")
    
    featureName = featureName.strip()
    if len(featureName) == 0:
        raise ValueError(f"{errorMessage}")
    
    return featureName

def get_groupIndex_from_featureNames(featureNames):
    # from a list of feature names, get the group index for each feature
    print("We convert binary feature names to continuous feature names\nNote that the continuous feature names should be in the form of 'FeatureName<=Threshold' or 'Threshold1<FeatureName<=Threshold2' or 'FeatureName_isNaN'!\nFor datasets from RiskSLIM (https://github.com/ustunb/risk-slim/tree/master/examples/data), we hardcode the conversion since the feature names do not follow the above format.")

    
    groupIndex = check_if_featureNames_come_from_RiskSLIM_GitHub_data(featureNames)
    if len(groupIndex) > 0:
        return np.asarray(groupIndex, dtype=int)
    
    continuousFeatureNameIndexDict = dict()
    numContinuousFeatures = 0

    for featureName in featureNames:
        continuousFeatureName = get_continuousFeatureName_from_binaryFeatureName(featureName)
        if continuousFeatureName not in continuousFeatureNameIndexDict:
            continuousFeatureNameIndexDict[continuousFeatureName] = numContinuousFeatures
            numContinuousFeatures += 1
        groupIndex.append(continuousFeatureNameIndexDict[continuousFeatureName])
    
    return np.asarray(groupIndex, dtype=int)

def check_if_featureNames_come_from_RiskSLIM_GitHub_data(featureNames):
    # check if feature names come from RiskSLIM GitHub data
    # if so, return the group index
    # if not, return an empty list
    
    adult_featureNames = [
        'Age_leq_21', 'Age_22_to_29', 'Age_30_to_44', 'Age_45_to_59', 'Age_geq_60',
        'EmployedInPrivateSector', 'EmployedInPublicSector', 'SelfEmployedNotInc', 'SelfEmployedInc',
        'HSDiploma', 'ProfVocOrAS', 'Bachelors', 'Graduate', 'NoHS',
        'JobManagerial', 'JobAdministrative', 'JobArmedForces', 'JobService', 'JobSkilledSpecialty', 'JobAgriculture',
        'Married', 'DivorcedOrSeparated', 'Widowed', 'NeverMarried',
        'WorkHrsPerWeek_lt_40', 'WorkHrsPerWeek_40_to_50', 'WorkHrsPerWeek_geq_50',
        'Male', 'Female',
        'White', 'Black', 'OtherRace', 'NativeUSorCanada', 'NativeImmigrant',
        'AnyCapitalGains', 'AnyCapitalLoss']
    adult_groupIndex = [
        0, 0, 0, 0, 0,  # Age-related features
        1, 1, 1, 1,  # Employment-related features
        2, 2, 2, 2, 2,  # Education-related features
        3, 3, 3, 3, 3, 3,  # Job-related features
        4, 4, 4, 4,  # Marital status-related features
        5, 5, 5,  # Work hours-related features
        6, 6,  # Gender-related features
        7, 7, 7, 7, 7, # Race-related features
        8, 8  # Native and capital gains/loss features
        ]

    bank_featureNames = [
        'age_lt_25', 'age_25_to_30', 'age_30_to_40', 'age_40_to_50', 'age_50_to_60', 'age_ge_60',
        'job_unemployed', 'job_retired', 'job_student', 'job_housemaid', 'job_management', 'job_entrepreneur', 'job_blue_collar', 'job_technician', 'job_admin', 'job_services',
        'married', 'single', 'divorced',
        'education_university', 'education_highschool', 'education_basic4yr', 'education_basic6yr', 'education_basic9yr', 'education_professional',
        'credit_is_or_may_be_in_default', 'has_housing_loan', 'has_personal_loan',
        'called_cellphone', 'called_landline',
        'call_in_Q1', 'call_in_Q2', 'call_in_Q3', 'call_in_Q4',
        'call_on_mon', 'call_on_tue', 'call_on_wed', 'call_on_thu', 'call_on_fri',
        'any_prior_calls_in_campaign', 'mult_prior_calls_in_campaign', 'called_before', 'any_prior_calls_b4_campaign', 'mult_prior_calls_b4_campaign', 'calls_b4_campaign_failed', 'calls_b4_campaign_successful',
        'emp_var_rate_lt_n1', 'emp_var_rate_gt_1',
        'nr_employed_lt_5100', 'nr_employed_gt_5100',
        'cons_price_idx_geq_93_5', 'cons_conf_idx_leq_n40',
        'euribor3m_geq_100', 'euribor3m_geq_200', 'euribor3m_geq_300', 'euribor3m_geq_400', 'euribor3m_geq_500']
    bank_groupIndex = [
        0, 0, 0, 0, 0, 0,  # Age-related features
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Job-related features
        2, 2, 2,  # Marital status-related features
        3, 3, 3, 3, 3, 3,  # Education-related features
        4, 4, 4,  # Credit-related features
        5, 5,  # Call device-related features
        6, 6, 6, 6,  # Call timing-related features (quarters)
        7, 7, 7, 7, 7,  # Call timing-related features (days)
        8, 8, 8, 8, 8, 8, 8,  # Call history-related features
        9, 9,  # Employment variable rate-related features
        10, 10,  # Number of employed-related features
        11, 11,  # Consumer price and confidence index-related features
        12, 12, 12, 12, 12  # Euribor rate-related features
    ]
    
    breastcancer_featureNames = ['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']
    breastcancer_groupIndex = [
        0,  # ClumpThickness (Cell thickness-related features)
        1,  # UniformityOfCellSize (Cell uniformity-related features)
        1,  # UniformityOfCellShape (Cell uniformity-related features)
        2,  # MarginalAdhesion (Adhesion-related features)
        3,  # SingleEpithelialCellSize (Epithelial cell size-related features)
        4,  # BareNuclei (Nuclei-related features)
        5,  # BlandChromatin (Chromatin-related features)
        6,  # NormalNucleoli (Nucleoli-related features)
        7   # Mitoses (Mitoses-related features)
    ]

    mammo_featureNames = [
        'RoundShape', 'OvalShape', 'LobularShape', 'IrregularShape', 
        'CircumscribedMargin', 'MicrolobulatedMargin', 'ObscuredMargin', 'IllDefinedMargin', 'SpiculatedMargin',
        'Density', 
        'Age_lt_30', 'Age_geq_30', 'Age_geq_45', 'Age_geq_60']
    mammo_groupIndex = [
        0, 0, 0, 0,  # Shape-related features
        1, 1, 1, 1, 1,  # Margin-related features
        2,  # Density-related features
        3, 3, 3, 3  # Age-related features
    ]

    mushroom_featureNames = [
        'cap_shape_eq_convex', 'cap_shape_eq_bell', 'cap_shape_eq_sunken', 'cap_shape_eq_flat', 'cap_shape_eq_knobbed', 'cap_shape_eq_conical',
        'cap_surface_eq_smooth', 'cap_surface_eq_scaly', 'cap_surface_eq_fibrous', 'cap_surface_eq_grooves',
        'cap_color_eq_brown', 'cap_color_eq_yellow', 'cap_color_eq_white', 'cap_color_eq_gray', 'cap_color_eq_red', 'cap_color_eq_pink', 'cap_color_eq_buff', 'cap_color_eq_purple', 'cap_color_eq_cinnamon', 'cap_color_eq_green',
        'bruises_eq_TRUE', 
        'odor_eq_pungent', 'odor_eq_almond', 'odor_eq_anise', 'odor_eq_none', 'odor_eq_foul', 'odor_eq_creosote', 'odor_eq_fishy', 'odor_eq_spicy', 'odor_eq_musty',
        'gill_attachment_eq_free', 'gill_attachment_eq_attached', 
        'gill_spacing_eq_close', 'gill_spacing_eq_crowded',
        'gill_size_eq_narrow', 'gill_size_eq_broad', 
        'gill_color_eq_black', 'gill_color_eq_brown', 'gill_color_eq_gray', 'gill_color_eq_pink', 'gill_color_eq_white', 'gill_color_eq_chocolate', 'gill_color_eq_purple', 'gill_color_eq_red', 'gill_color_eq_buff', 'gill_color_eq_green', 'gill_color_eq_yellow', 'gill_color_eq_orange',
        'stalk_shape_eq_elarging', 'stalk_shape_eq_tapering',
        'stalk_root_eq_equal', 'stalk_root_eq_club', 'stalk_root_eq_bulbous', 'stalk_root_eq_rooted', 
        'stalk_surface_above_ring_eq_smooth', 'stalk_surface_above_ring_eq_fibrous', 'stalk_surface_above_ring_eq_grooves', 'stalk_surface_above_ring_eq_scaly', 
        'stalk_surface_below_ring_eq_smooth', 'stalk_surface_below_ring_eq_fibrous', 'stalk_surface_below_ring_eq_scaly', 'stalk_surface_below_ring_eq_grooves',
        'stalk_color_above_ring_eq_white', 'stalk_color_above_ring_eq_gray', 'stalk_color_above_ring_eq_pink', 'stalk_color_above_ring_eq_brown', 'stalk_color_above_ring_eq_buff', 'stalk_color_above_ring_eq_red', 'stalk_color_above_ring_eq_orange', 'stalk_color_above_ring_eq_cinnamon', 'stalk_color_above_ring_eq_yellow',
        'stalk_color_below_ring_eq_white', 'stalk_color_below_ring_eq_pink', 'stalk_color_below_ring_eq_gray', 'stalk_color_below_ring_eq_buff', 'stalk_color_below_ring_eq_brown', 'stalk_color_below_ring_eq_red', 'stalk_color_below_ring_eq_yellow', 'stalk_color_below_ring_eq_orange', 'stalk_color_below_ring_eq_cinnamon',
        'veil_color_eq_white', 'veil_color_eq_brown', 'veil_color_eq_orange', 'veil_color_eq_yellow', 
        'ring_number_eq_1', 'ring_number_eq_2',
        'ring_type_eq_pendant', 'ring_type_eq_evanescent', 'ring_type_eq_large', 'ring_type_eq_flaring', 'ring_type_eq_none',
        'spore_print_color_eq_black', 'spore_print_color_eq_brown', 'spore_print_color_eq_purple', 'spore_print_color_eq_chocolate', 'spore_print_color_eq_white', 'spore_print_color_eq_green', 'spore_print_color_eq_orange', 'spore_print_color_eq_yellow', 'spore_print_color_eq_buff',
        'population_eq_scattered', 'population_eq_numerous', 'population_eq_abundant', 'population_eq_several', 'population_eq_solitary', 'population_eq_clustered',
        'habitat_eq_urban', 'habitat_eq_grasses', 'habitat_eq_meadows', 'habitat_eq_woods', 'habitat_eq_paths', 'habitat_eq_waste', 'habitat_eq_leaves']
    mushroom_groupIndex = [
        0, 0, 0, 0, 0, 0,  # Cap shape-related features
        1, 1, 1, 1,  # Cap surface-related features
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # Cap color-related features
        3,  # Bruises-related feature
        4, 4, 4, 4, 4, 4, 4, 4, 4,  # Odor-related features
        5, 5,  # Gill attachment-related features
        6, 6,  # Gill spacing-related features
        7, 7,  # Gill size-related features
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,  # Gill color-related features
        9, 9,  # Stalk shape-related features
        10, 10, 10, 10,  # Stalk root-related features
        11, 11, 11, 11,  # Stalk surface above ring-related features
        12, 12, 12, 12,  # Stalk surface below ring-related features
        13, 13, 13, 13, 13, 13, 13, 13, 13,  # Stalk color above ring-related features
        14, 14, 14, 14, 14, 14, 14, 14, 14,  # Stalk color below ring-related features
        15, 15, 15, 15,  # Veil color-related features
        16, 16,  # Ring number-related features
        17, 17, 17, 17, 17,  # Ring type-related features
        18, 18, 18, 18, 18, 18, 18, 18, 18,  # Spore print color-related features
        19, 19, 19, 19, 19, 19,  # Population-related features
        20, 20, 20, 20, 20, 20, 20  # Habitat-related features
    ]

    spambase_featureNames = [
        'WordFreqMake', 'WordFreqAddress', 'WordFreqAll', 'WordFreq3D', 'WordFreqOur', 'WordFreqOver', 'WordFreqRemove', 'WordFreqInternet', 'WordFreqOrder', 'WordFreqMail', 'WordFreqReceive', 'WordFreqWill', 'WordFreqPeople', 'WordFreqReport', 'WordFreqAddresses', 'WordFreqFree', 'WordFreqBusiness', 'WordFreqEmail', 'WordFreqYou', 'WordFreqCredit', 'WordFreqYour', 'WordFreqFont', 'WordFreq0', 'WordFreqMoney', 'WordFreqHP', 'WordFreqHPL', 'WordFreqGeorge', 'WordFreq650', 'WordFreqLab', 'WordFreqLabs', 'WordFreqTelnet', 'WordFreq857', 'WordFreqData', 'WordFreq415', 'WordFreq85', 'WordFreqTechnology', 'WordFreq1999', 'WordFreqParts', 'WordFreqPM', 'WordFreqDirect', 'WordFreqCS', 'WordFreqMeeting', 'WordFreqOriginal', 'WordFreqProject', 'WordFreqRE', 'WordFreqEDU', 'WordFreqTable', 'WordFreqConference',
        'CharFreqSemicolon', 'CharFreqParentheses', 'CharFreqBracket', 'CharFreqExcalamationMark', 'CharFreqDollarSign', 'CharFreqPound',
        'CapitalRunLengthAverage', 'CapitalRunLengthLongest', 'CapitalRunLengthTotal']
    spambase_groupIndex = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # Word frequency-related features
        1, 1, 1, 1, 1, 1,  # Character frequency-related features
        2, 2, 2  # Capital run length-related features
    ]

    groupIndex = []
    if featureNames == adult_featureNames:
        groupIndex = adult_groupIndex
    elif featureNames == bank_featureNames:
        groupIndex = bank_groupIndex
    elif featureNames == breastcancer_featureNames:
        groupIndex = breastcancer_groupIndex
    elif featureNames == mammo_featureNames:
        groupIndex = mammo_groupIndex
    elif featureNames == mushroom_featureNames:
        groupIndex = mushroom_groupIndex
    elif featureNames == spambase_featureNames:
        groupIndex = spambase_groupIndex

    return groupIndex