import pandas as pd

df1=pd.read_csv('shared_articles.csv')
df2=pd.read_csv('users_interactions.csv')

# print(df1.head)
# print(df2.head)

# ________________________________________________

# DEMOGRAPHIC FILTERING

df1=df1[df1['eventType'] == 'CONTENT SHARED']
# print(df1.head)

def totalEvents(df1_row):
    total_views= df2[(df2["contentId"]== df1_row['contentId']) & (df2['eventType']=='VIEW')].shape[0]
    total_likes= df2[(df2["contentId"]== df1_row['contentId']) & (df2['eventType']=='LIKE')].shape[0]
    total_bookmarks= df2[(df2["contentId"]== df1_row['contentId']) & (df2['eventType']=='BOOKMARK')].shape[0]
    total_follows= df2[(df2["contentId"]== df1_row['contentId']) & (df2['eventType']=='FOLLOW')].shape[0]
    total_comments= df2[(df2["contentId"]== df1_row['contentId']) & (df2['eventType']=='COMMENT CREATED')].shape[0]
    return total_views+total_likes+total_bookmarks+total_follows+total_comments

df1['total_events']=df1.apply(totalEvents,axis=1)
df1.head()

# _________________________________________________

# CONTENT BASED FILTERING

def to_lower(x):
    if isinstance(x,str):
        return x.lower()
    else:
        return ''

df1['title']=df1['title'].apply(to_lower)
df1.head()

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df1['title'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df1 = df1.reset_index()
indices = pd.Series(df1.index, index=df1['contentId'])

def getRecommendations(contentId, cosine_sim):
    idx = indices[contentId]
    simScores = list(enumerate(cosine_sim[idx]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
    simScores = simScores[1:11]
    articleIndices = [i[0] for i in simScores]
    return df1['contentId'].iloc[articleIndices]

getRecommendations(-133139342397538859, cosine_sim2)




