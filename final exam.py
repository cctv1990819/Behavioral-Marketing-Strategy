# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:59:43 2019

@author: Kaiyi Zou

C:\Users\Administrator\Desktop\Hult\Machine learning\Individual assignment\EXAM

Final exam
"""
# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

# import data: survey_df
file = 'finalExam_Mobile_App_Survey_Data.xlsx'
 
survey_df = pd.read_excel(file)

# EDA
survey_df.head(5)

survey_df.info()


# check variance
print(pd.np.var(survey_df))

##############################################################################
# Principal Components analysis
##############################################################################
# prepare data for analysis
analysis_df = survey_df.copy()

analysis_df = analysis_df.drop('caseID', 
                               axis = 1) # caseID is not helpful for analysis

# drop demographic data: q1 and columns after q48
analysis_df = analysis_df.drop(['q1',
                                'q48',
                                'q49',
                                'q50r1',
                                'q50r2',
                                'q50r3',
                                'q50r4',
                                'q50r5',
                                'q54',
                                'q55',
                                'q56',
                                'q57'],axis = 1)


# create scaler
scaler = StandardScaler()

#fit and transform analysis_df
scaled_data = scaler.fit_transform(analysis_df)

# save analysis_df as data frame
scaled_df = pd.DataFrame(scaled_data)

# add column name to scaled_df
scaled_df.columns = analysis_df.columns

# check variance
print(pd.np.var(scaled_df))

# look the coorelation
analysis_corr = scaled_df.corr().round(2)

# visulize coorelation
fig, ax = plt.subplots(figsize = (12, 12))
sns.heatmap(analysis_corr,
            cmap = "coolwarm",
            square = True)
plt.show()

'''
columns with relatively high correlation with others:
    q4r6,
    q11,
    q13r1,
    q24r3,
    q25r1,
    q25r4,
    q25r6,
    q26r18
'''
# subset scaled_df base on correlation, get useful_df
usefull_df = scaled_df.loc[:,['q4r6',
                              'q11',
                              'q13r1',
                              'q24r3',
                              'q25r1',
                              'q25r4',
                              'q25r6',
                              'q26r18']]

#####################
# find the minimum features needed
# create PCA for all features: full_pca
full_pca = PCA(n_components = None, 
               random_state = 508)

# fit full_pca with survey_df
full_pca.fit(usefull_df)

# transform scaled_df
full_pca.transform(usefull_df)

# plot the explained variances to find the minimum principal components
features = range(full_pca.n_components_)
plt.figure(figsize=(14, 10))
plt.plot(features, 
        full_pca.explained_variance_ratio_,
        linewidth = 2,
        marker = 'o',
        markersize = 7,
        markeredgecolor = 'black',
        markerfacecolor = 'grey')
plt.xlabel("PCA Features")
plt.ylabel("Variance")
plt.xticks(features, rotation = 90)
plt.show()


print(f'''
      In order to explain at least 80% variance of the data set, the principal 
components we need are list:
\n 5 principal components: {(full_pca.explained_variance_ratio_[0]
                            + full_pca.explained_variance_ratio_[1]
                            + full_pca.explained_variance_ratio_[2]
                            + full_pca.explained_variance_ratio_[3]
                            + full_pca.explained_variance_ratio_[4]).round(2)}
      ''')

#####################
# build pca with 5 principal components
    
# PCA with 26 principal components
pca_5 = PCA(n_components = 5,
             random_state = 508)    
    
# fit scaled_df
pca_5.fit(usefull_df)

# transform scaled_df and store as data frame: pca_df
pca_df = pd.DataFrame(pca_5.transform(usefull_df))

#####################
#Analyze factor loadings to understand principal 
factor_loadings_df = pd.DataFrame(pd.np.transpose(pca_5.components_))


factor_loadings_df = factor_loadings_df.set_index(usefull_df.columns[:])


# rename pca_df base on their behavior 
'''
group 0: old_follower
            they don't use social networking apps and more like a follower than leader
group 1: social_follower
            they are on social networking apps but more like a follower than leader
group 2: single_rich
            they don't have many apps on their phone and more like to follow oeders
group 3: poor_leader
            they don't have many apps on their phone, but they have their own ideas.
            more like a leader. but they don't addict to luxury goods.
group 4: influencer 
            they use a lot of apps. more like a leader.they don't addict to luxury goods.
'''
pca_df.columns = ['old_follower',
                  'social_follower',
                  'single_rich',
                  'poor_leader',
                  'influencer']


#####################
# market need analysis
market_analysis_df = pd.concat([pca_df, survey_df.loc[:,['q1',
                                                        'q48',
                                                        'q49',
                                                        'q50r1',
                                                        'q50r2',
                                                        'q50r3',
                                                        'q50r4',
                                                        'q50r5',
                                                        'q54',
                                                        'q55',
                                                        'q56',
                                                        'q57']]], axis = 1)
    
# rename q1: age
age = {1 : 'Under 18',
       2 : '18-24',
       3 : '25-29',
       4 : '30-34',
       5 : '35-39',
       6 : '40-44',
       7 : '45-49',
       8 : '50-54',
       9 : '55-59',
       10: '60-64',
       11: '65 or over'}

market_analysis_df['q1'].replace(age, inplace = True)

# rename q48: education
education = {1 : 'Some high school',
             2 : 'High school graduate',
             3 : 'Some college',
             4 : 'College graduate',
             5 : 'Some post-graduate studies',
             6 : 'Post graduate degree'}

market_analysis_df['q48'].replace(education, inplace = True)

# rename q49: marital_status
marital_status = {1 : 'Married',
                  2 : 'Single',
                  3 : 'Single with a partner',
                  4 : 'Separated/Widowed/Divorced'}

market_analysis_df['q49'].replace(marital_status, inplace = True)

# create new column: q50 to store information in q50r1,q50r2,q50r3,q50r4,q50r5
for index in range(len(market_analysis_df)):
    # if q50r1 = 1, q50 should be No children
    if market_analysis_df.loc[index, 'q50r1'] == 1:
        market_analysis_df.loc[index, 'q50'] = 'No children'
    # if q50r2 = 1, q50 should be Yes, children under 6 years old
    elif market_analysis_df.loc[index, 'q50r2'] == 1:
        market_analysis_df.loc[index, 'q50'] = 'Yes, children under 6 years old'
    # if q50r3 = 1, q50 should be Yes, children 6-12 years old
    elif market_analysis_df.loc[index, 'q50r3'] == 1:
        market_analysis_df.loc[index, 'q50'] = 'Yes, children 6-12 years old'
    # if q50r4 = 1, q50 should be Yes, children 13-17 years old
    elif market_analysis_df.loc[index, 'q50r4'] == 1:
        market_analysis_df.loc[index, 'q50'] = 'Yes, children 13-17 years old' 
    # if q50r5 = 1, q50 should be Yes, children 18 or older
    elif market_analysis_df.loc[index, 'q50r5'] == 1:
        market_analysis_df.loc[index, 'q50'] = 'Yes, children 18 or older'

# drop q50r1,q50r2,q50r3,q50r4,q50r5    
market_analysis_df = market_analysis_df.drop(['q50r1',
                                              'q50r2',
                                              'q50r3',
                                              'q50r4',
                                              'q50r5'],axis = 1)

# rename q54: race
race = {1 : 'White or Caucasian',
        2 : 'Black or African American',
        3 : 'Asian',
        4 : 'Native Hawaiian or Other Pacific Islander',
        5 : 'American Indian or Alaska Native',
        6 : 'Other race'}

market_analysis_df['q54'].replace(race, inplace = True)

# rename q55: Hispanic or Latino
HL = {1 : 'Hispanic or Latino',
      2 : 'Not Hispanic or Latino'}

market_analysis_df['q55'].replace(HL, inplace = True)

# remane q56: income
income = {1 : 'Under $10,000',
          2 : '$10,000-$14,999',
          3 : '$15,000-$19,999',
          4 : '$20,000-$29,999',
          5 : '$30,000-$39,999',
          6 : '$40,000-$49,999',
          7 : '$50,000-$59,999',
          8 : '$60,000-$69,999',
          9 : '$70,000-$79,999',
          10: '$80,000-$89,999',
          11: '$90,000-$99,999',
          12: '$100,000-$124,999',
          13: '$125,000-$149,999',
          14: '$150,000 and over'}

market_analysis_df['q56'].replace(income, inplace = True)

# rename q57: gender: q57
gender = {1 : 'Male',
          2 : 'Female'}

market_analysis_df['q57'].replace(gender, inplace = True)

# Analyzing by gender
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q57',
               y = 'social_follower',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])

plt.subplot(2,1,2)
sns.violinplot(x = 'q57',
               y = 'influencer',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.tight_layout()
plt.show()

'''
gender doesn't make much difference in different market
'''
# Analyzing by age: q1
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q1',
               y = 'social_follower',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.title('Time is Luxury\nFigure 3')

plt.subplot(2,1,2)
sns.violinplot(x = 'q1',
               y = 'influencer',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()

'''
social follower: older people are more likely to buy
influncer: all age groups are at or bolow average
'''

# Analyzing by education: q48
fig, ax = plt.subplots(figsize = (12, 10))
plt.subplot(2,1,1)
sns.violinplot(x = 'q48',
               y = 'social_follower',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q48',
               y = 'influencer',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()
'''
no much difference
'''
# Analyzing by marital status: q49
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q49',
               y = 'social_follower',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q49',
               y = 'influencer',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()
'''
separated people from social follower group is more likely to buy
'''
# Analyzing by Race: q54
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q54',
               y = 'social_follower',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q54',
               y = 'influencer',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()
'''
white or caucasian from social follower
asian from influencer
'''
# Analyzing by income: q56
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q56',
               y = 'social_follower',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.title('Likelyhood of Purchase across Income Level\nfigure1')

plt.subplot(2,1,2)
sns.violinplot(x = 'q56',
               y = 'influencer',
               data = market_analysis_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()
'''
20,000-29,999/40,000-49,999 from influencer
125,000-149,999/60,000-69,999/40,000-49,999 from social follower
'''

##############################################################################
# KMeans
##############################################################################
# check variance again
print(pd.np.var(pca_df))

# scale again, store as data frame: pca_scaled
scaler = StandardScaler()

kmean_scaled = scaler.fit_transform(pca_df)

kmean_scaled = pd.DataFrame(kmean_scaled)

# add columns to kmean_scaled
kmean_scaled.columns = pca_df.columns

# check variance
print(pd.np.var(kmean_scaled))

#####################
# evaluate clusters with inertia

# plot inertia
nc = range(1,50)
inertias = []

for k in nc:
    # build KMean model
    model = KMeans(n_clusters = k)
    # fit the model
    model.fit(kmean_scaled)
    # append intertia of each k to list: inertias
    inertias.append(model.inertia_)
    
fig, ax = plt.subplots(figsize = (12,8))
plt.plot(nc, inertias, '-o')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(nc)
plt.show()

#####################
# divide into 6 clusters
clusters_6 = KMeans(n_clusters = 6,
                    random_state = 508)

# fit clusters_6
clusters_6.fit(kmean_scaled)

kmeans_output = pd.DataFrame({'cluster': clusters_6.labels_})

# combine with pca
kmeans_pca = pd.concat([kmeans_output, kmean_scaled], axis = 1)

# Reattach demographic information
final_df = pd.concat([kmeans_pca,survey_df.loc[:,['q1',
                                                  'q48',
                                                  'q49',
                                                  'q50r1',
                                                  'q50r2',
                                                  'q50r3',
                                                  'q50r4',
                                                  'q50r5',
                                                  'q54',
                                                  'q55',
                                                  'q56',
                                                  'q57']]], axis = 1)
    

# Output result to Excel file
writer = pd.ExcelWriter('output/market_need.xlsx')
final_df.to_excel(writer, 'Customer Segments', header=True)
pca_df.to_excel(writer, 'Market Need', header=True)
writer.save()

#####################
# market segment analysis
# rename q1: age
final_df['q1'].replace(age, inplace = True)

# rename q48: education
final_df['q48'].replace(education, inplace = True)

# rename q49: marital_status
final_df['q49'].replace(marital_status, inplace = True)

# create new column: q50 to store information in q50r1,q50r2,q50r3,q50r4,q50r5
for index in range(len(final_df)):
    # if q50r1 = 1, q50 should be No children
    if final_df.loc[index, 'q50r1'] == 1:
        final_df.loc[index, 'q50'] = 'No children'
    # if q50r2 = 1, q50 should be Yes, children under 6 years old
    elif final_df.loc[index, 'q50r2'] == 1:
        final_df.loc[index, 'q50'] = 'Yes, children under 6 years old'
    # if q50r3 = 1, q50 should be Yes, children 6-12 years old
    elif final_df.loc[index, 'q50r3'] == 1:
        final_df.loc[index, 'q50'] = 'Yes, children 6-12 years old'
    # if q50r4 = 1, q50 should be Yes, children 13-17 years old
    elif final_df.loc[index, 'q50r4'] == 1:
        final_df.loc[index, 'q50'] = 'Yes, children 13-17 years old' 
    # if q50r5 = 1, q50 should be Yes, children 18 or older
    elif final_df.loc[index, 'q50r5'] == 1:
        final_df.loc[index, 'q50'] = 'Yes, children 18 or older'

# drop q50r1,q50r2,q50r3,q50r4,q50r5    
final_df = final_df.drop(['q50r1',
                          'q50r2',
                          'q50r3',
                          'q50r4',
                          'q50r5'],axis = 1)

# rename q54: race
final_df['q54'].replace(race, inplace = True)

# rename q55: Hispanic or Latino
final_df['q55'].replace(HL, inplace = True)

# remane q56: income
final_df['q56'].replace(income, inplace = True) 

# rename q57 : gender
final_df['q57'].replace(gender, inplace = True)

#####################
# plotting
# Analyzing by gender: q57
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q57',
               y = 'social_follower',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.axhline(linewidth=2, color='grey')
plt.title('All Equality Across Gender \nFigure 2')

plt.subplot(2,1,2)
sns.violinplot(x = 'q57',
               y = 'influencer',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()
'''
Customer segment: 
    Social Follower: cluster 3 and4
    influencer: cluster 2 and 3
gender doesn't matter
'''

# Analyzing by age: q1
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q1',
               y = 'social_follower',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q1',
               y = 'influencer',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()

'''
all age group performe the same way in different market
Customer segment: 
    Old follower: cluster 1
    Social Follower: cluster 4
    single rich: cluster 2 and 4
    poor leader: cluster 5
    influencer: cluster 2 and 3

'''

# Analyzing by education: q48
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q48',
               y = 'social_follower',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q48',
               y = 'influencer',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()

# Analyzing by marital status: q49
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q49',
               y = 'social_follower',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q49',
               y = 'influencer',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()

# Analyzing by Race: q54
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q54',
               y = 'social_follower',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q54',
               y = 'influencer',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()

# Analyzing by income： q56
fig, ax = plt.subplots(figsize = (12, 10))

plt.subplot(2,1,1)
sns.violinplot(x = 'q56',
               y = 'social_follower',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Social Follower')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')

plt.subplot(2,1,2)
sns.violinplot(x = 'q56',
               y = 'influencer',
               hue = 'cluster',
               data = final_df)
plt.xlabel('')
plt.ylabel('Influencer')
plt.yticks([-6,-4,-2,0,2,4,6])
plt.xticks(rotation = 60)
plt.axhline(linewidth=2, color='grey')
plt.tight_layout()
plt.show()

