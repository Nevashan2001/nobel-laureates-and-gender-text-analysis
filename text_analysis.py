import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('vader_lexicon')
from collections import Counter
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

df_nobel = pd.read_csv("/Users/nevashanbaskaran/somenobel.csv")
## 1.1 ^
print(df_nobel.head())

laureates = df_nobel['author_name'].unique()
print(len(laureates))
# 7
for name in laureates:
    print(name)
# Joshua Angrist
# Philippe Aghion
# Daron Acemoglu
# Ben Bernanke
# David Card
# Claudia Goldin
# Esther Duflo
## 1.2 7 Nobel laureates. Joshua Angrist: Won for figuring out how to use "natural experiments" to find real-world cause and effect (like how more education causes higher pay), 
##Philippe Aghion: Won for explaining that economic growth is driven by "creative destruction," where new, better inventions replace old ones, 
##Daron Acemoglu: Won for showing that countries are rich or poor mainly because of their political and economic "institutions" (like having good laws vs. corruption), 
##Ben Bernanke: Won for his research on the Great Depression, proving that bank failures were a cause of the long crisis, not just a result of it, 
##David Card: Won for using real-world data to show that raising the minimum wage doesn't necessarily cut jobs and that immigration doesn't lower wages for native workers, 
##Claudia Goldin: Won for creating the first complete history of women's work and pay, explaining the reasons for the gender pay gap over the centuries, 
##Esther Duflo: Won for pioneering the use of real-world experiments (like medical trials) to find the most effective ways to reduce global poverty.
earliest_year = df_nobel['year'].min()
earliest_publication = df_nobel[df_nobel['year'] == earliest_year]
print(earliest_publication)
#          author_name                                              title  year
# 52   Philippe Aghion  ON THE GENERIC INEFFICIENCY OF DIFFERENTIABLE ...  1985
# 119     Ben Bernanke  ADJUSTMENT COSTS, DURABLES, AND AGGREGATE CONS...  1985
# 362       David Card  USING THE LONGITUDINAL STRUCTURE OF EARNINGS T...  1985
## 1.3 The laureates with the earliest publication are Philippe Aghion, Ben Bernanke, David Card all in 1985.
punctuation = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
stopwords_set = set(stopwords.words('english'))
def clean(words):
    lower = words.lower()
    punc_free = ''.join(ch for ch in lower if ch not in punctuation)
    numb_free = ''.join(ch for ch in punc_free if not ch.isdigit())    
    word_list = numb_free.split()
    stop_free = [word for word in word_list if word not in stopwords_set]
    return ' '.join(stop_free)

df_nobel['cleaned'] = df_nobel['title'].apply(clean)
print(df_nobel[['title', 'cleaned']].head())
## 1.4 ^
for name in laureates:
        name_df = df_nobel[df_nobel['author_name'] == name]
        all_titles = ' '.join(name_df['cleaned'])
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(all_titles)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f"Word Cloud for {name}")
        plt.show()
## 1.5 Yes after generating each word cloud the main topics in each of the clouds aligns well with what each laureate won their Nobel Prize for. 
##Angrist's most popular words are school and effect which makes sense as he won the prize for research on cause and effect using schools as examples, 
##Aghlon's most common words are growth and innovation because he won his prize for explaining how growth happens due to new inventions replacing old ones which is innovation, 
##Acemoglu's most used words are policy, democracy, etc which is because he shows countries prosper economically because of their instuttions
##Bernanke's most popular words are framework, inflantion, policy which makes sense as he did research on the great depression
##Card's most common words are wage,immigration and effect which is because he researched minimum wage, and the effect of immigration on jobs
##Goldin's most used words are women, education and employment because she work on researched related to the gender pay gap
##Duflo's most common words are evidence, expierment and ranomized because he focused on real-world experiments to reduce poverty
def three_word_clouds(author_name):
    author_df = df_nobel[df_nobel['author_name'] == author_name]
    pre_2000 = ' '.join(author_df[author_df['year'] <= 2000]['cleaned'])
    mid_2000 = ' '.join(author_df[(author_df['year'] >= 2001) & (author_df['year'] <= 2010)]['cleaned'])
    post_2010 = ' '.join(author_df[author_df['year'] > 2010]['cleaned'])
    plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.title("Before or in 2000")
    if  pre_2000.strip():
        wordcloud1 = WordCloud(background_color='white').generate(pre_2000)
        plt.imshow(wordcloud1)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("2001-2010")
    if mid_2000.strip():
        wordcloud2 = WordCloud(background_color='white').generate(mid_2000)
        plt.imshow(wordcloud2)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("After 2010")
    if post_2010.strip():
        wordcloud3 = WordCloud(background_color='white').generate(post_2010)
        plt.imshow(wordcloud3)
    plt.axis("off")

    plt.show()
## 1.6^
three_word_clouds('Daron Acemoglu')
three_word_clouds('Ben Bernanke')
three_word_clouds('Claudia Goldin')
## 1.7 Daron Acemoglu : Yes there is change in research focus as he focuses on the theoretical and market before 2010 and after 2010 he is much more focus on growth, innovation as well as techonlogy and networks. Ben Bernanke: No his research focus seems to stay consistent as he focuses on economic policies and government related economics the entire time. Claudia Goldin: No she does not show much change in research focus as she seems to focus on women's role in economics throughout all her research with words such as female and women being very dominant.
df_classifier = pd.read_excel("/Users/nevashanbaskaran/female_male_classifiers.xlsx")
df_post = pd.read_csv("/Users/nevashanbaskaran/gendered_posts.csv")
##2.1 ^
female_words = df_classifier[df_classifier['female'] == 1]['word']
male_words = df_classifier[df_classifier['female'] == 0]['word']
female_classifiers = set(female_words)
male_classifiers = set(male_words)
print(female_classifiers)
print(male_classifiers)
print(len(female_classifiers))
# 57
print(len(male_classifiers))
# 236
## 2.2 The female classifier data contains words that only refer to women and would usually considered a feminine name whereas the male classifier data contains words that only refer to men and names that are tradiationally considered masculine. There are 57 female classifiers and 236 male classifiers.
def clean_post(words):
    lower = words.lower()
    punc_free = ''.join(ch for ch in lower if ch not in punctuation)
    numb_free = ''.join(ch for ch in punc_free if not ch.isdigit())
    return numb_free.split()
df_post['cleaned_post'] = df_post['raw_post'].apply(clean_post)
## 2.3 ^
def check_gender(word_list, classifiers):
    return not set(word_list).isdisjoint(classifiers)

df_post['is_female'] = df_post['cleaned_post'].apply(lambda x: check_gender(x, female_classifiers))
df_post['is_male'] = df_post['cleaned_post'].apply(lambda x: check_gender(x, male_classifiers))
print(len(df_post[(df_post['is_female'] == True) & (df_post['is_male'] == False)]))
# 87605
print(len(df_post[(df_post['is_male'] == True) & (df_post['is_female'] == False)]))
# 303450
## 2.4 There are 87605 posts that refer exclusively to women and 303450 posts that refer exclusively to men
stopwords_post = stopwords_set.union(female_classifiers).union(male_classifiers)
def remove_all_stops(word_list):
    return [ch for ch in word_list if ch not in stopwords_post]
female_posts_df = df_post[(df_post['is_female'] == True) & (df_post['is_male'] == False)].copy()
male_posts_df = df_post[(df_post['is_male'] == True) & (df_post['is_female'] == False)].copy()
female_words = [w for post in female_posts_df['cleaned_post'].apply(remove_all_stops) for w in post]
male_words = [w for post in male_posts_df['cleaned_post'].apply(remove_all_stops) for w in post]
female_count = Counter(female_words)
male_count = Counter(male_words)
print(female_count.most_common(50))
#[('like', 10687), ('dont', 8398), ('get', 8335), ('would', 7673), ('one', 6790), ('im', 6196), ('good', 6184), ('think', 5388), ('time', 5192), ('people', 5155), ('know', 5059), ('want', 4611), ('go', 4313), ('even', 4285), ('really', 4141), ('op', 3986), ('make', 3903), ('also', 3834), ('much', 3790), ('work', 3307), ('life', 3183), ('job', 3177), ('years', 3140), ('shes', 3099), ('way', 3027), ('better', 3006), ('going', 2988), ('see', 2981), ('never', 2974), ('could', 2871), ('say', 2855), ('well', 2851), ('first', 2802), ('us', 2797), ('got', 2697), ('year', 2681), ('love', 2663), ('youre', 2588), ('hot', 2560), ('still', 2532), ('need', 2521), ('find', 2486), ('take', 2460), ('many', 2450), ('said', 2428), ('back', 2397), ('thats', 2356), ('right', 2327), ('something', 2317), ('doesnt', 2297)]
print(male_count.most_common(50))
#[('like', 35896), ('one', 30218), ('would', 29776), ('good', 26725), ('people', 26004), ('dont', 25870), ('get', 25399), ('think', 22190), ('know', 20389), ('even', 19504), ('time', 18536), ('also', 17082), ('im', 16016), ('really', 15434), ('us', 15373), ('much', 15341), ('work', 14829), ('paper', 14215), ('top', 13434), ('years', 13181), ('make', 12741), ('go', 12557), ('want', 12113), ('well', 12042), ('see', 11976), ('economics', 11911), ('way', 11695), ('year', 11465), ('first', 11436), ('many', 11383), ('could', 11381), ('op', 11166), ('better', 11126), ('say', 11031), ('job', 10718), ('new', 10671), ('going', 10653), ('said', 10648), ('still', 10441), ('got', 10433), ('need', 9685), ('right', 9510), ('school', 9305), ('never', 9289), ('hes', 9111), ('research', 8999), ('great', 8951), ('market', 8878), ('phd', 8865), ('two', 8730)]
## 2.5 The patterns I can observe from the top 50 most frequently used words about both genders is that both lists seem to share a lot of commonly used informal "internet slang" such as like, don't, get, etc. However where they differ is that the posts about men much more commonly contain words related to work and research such research, phd, paper, etc whereas the posts about women much more commonly contain words regarding personal life and appareance such as love, hot, need, etc.
female_top200 = set([word for word, count in female_count.most_common(200)])
male_top200 = set([word for word, count in male_count.most_common(200)])
common_words = female_top200.intersection(male_top200)
new_stopwords = stopwords_post.union(common_words).union({"hes", "wants", "shes", "sure", "tell"})
def remove_new_stops(word_list):
    return [w for w in word_list if w not in new_stopwords]

female_string = ' '.join([w for post in female_posts_df['cleaned_post'].apply(remove_new_stops) for w in post])
male_string = ' '.join([w for post in male_posts_df['cleaned_post'].apply(remove_new_stops) for w in post])
wordcloud_female = WordCloud(background_color='white', width=800, height=400).generate(female_string or "No Data")
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_female)
plt.axis("off")
plt.title("Female Posts")
plt.show()
wordcloud_male = WordCloud(background_color='white', width=800, height=400).generate(male_string or "No Data")
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_male)
plt.axis("off")
plt.title("Male Posts")
plt.show()
## 2.6 ^^
sentiment_analyzer = SentimentIntensityAnalyzer()
def get_compound(text):
    return sentiment_analyzer.polarity_scores(str(text))['compound']
df_post['vader_score'] = df_post['raw_post'].apply(get_compound)
female_scores_df = pd.DataFrame({'vader_score': df_post.loc[female_posts_df.index, 'vader_score'], 'group': 'Female'})
male_scores_df = pd.DataFrame({'vader_score': df_post.loc[male_posts_df.index, 'vader_score'], 'group': 'Male'})
df_plot = pd.concat([female_scores_df, male_scores_df])
plt.figure(figsize=(10, 6))
sns.kdeplot(x='vader_score', hue='group', data=df_plot, shade=True, common_norm=False, linewidth=2)
plt.title("Density of Sentiment Scores")
plt.show()
## 2.7 ^^
## 2.8 In conclusion from this data it shows that in academia that men are much more likely to be discussed about for their achievements and profession with their wordcloud having many more professional terms whereas women are much more likely to be discussed about with words related to their physical appearance and personality instead such as hot. The sentiment anaylsis plot also shows the women are talked about slightly more negatively and slightly less positvely then men whereas men have a much higher peak at 0 which means a larger variety of words are being used that are exactly neutral or not in the VADER dictionary.
