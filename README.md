# Media-Bias-Prediction-Model
## Abstract
INSERT ABSTRACT HERE


## How to use the Model
The model is broken into several different .ipynb files. To do any type of analysis only the Project_Pipeline.ipynb folder needs to be used. If you aren't making any code changes and just doing analysis there is no need to go into the other files. Therefore, this readme is going to go into depth on only the pipeline.

### Installs
Run on initial startup.
The very first code block of Project_Pipeline.ipynb includes all the installations one should need to use the notebook. You can run the entire code block and it will auto install for you. You can see this code block below:
```
# Dataframe and sentiment analysis
!pip install pandas
!pip install spacy
!pip install spacytextblob
!python -m spacy download en_core_web_sm

# Webscraping
!pip install newspaper3k

# Topic modeling 
!pip install gensim

# Data vis
!pip install plotly
!pip install sklearn
!pip install matplotlib
!pip install wordcloud
!pip install seaborn
```
As you can see the installations are already nicely packaged into the category in which they belong. Here is a little more information on the details of the installs:

Pandas: Stores the data as a data frame table

Spacy: Used for NLP and has the machine learning module
    
SpacyTextBlob: Used for the sentiment analysis
    
NewsPaper: Used for web scraping

Gensim: Used for topic modeling

Plotly: Used for data visualization

Sklearn: Used for clustering and data visualization

Matplotlib: Used for clustering and data visualization

Wordcloud: Used for creating and displaying the word clouds

Seaborn: Used for displaying data visualizations

### Imports
Run on every startup.
After our installs code block we have our imports. If all the installs installed successfully this code block should run without issue. This code block is used to import all the dependencies for the model.

### Connect Helper Files
Run on every startup.
This code block is used to connect all the helper files to the main pipeline. This code block allows for the pipeline to be condensed and easy to read/use.

### Side Note
The sections from this point down will need to be run every time you want to analyze a new set of articles or change settings. Installs only needs to be ran the first time you use this project. Imports and Connect Helper Files needs to be ran when you first open the file but does not need to be ran every time you try a new set of articles or change settings.

### General Pipeline Settings, and Pre-Analysis Setup
This code block is where you can change the settings of the web scrapping and topic analysis. In addition, this is also where you put the URL to the .csv file you want scraped. We will walk through each setting in this section.

csv_file : This setting is the actual .csv file to be scrapped. It is important that this file is stored within the same folder as the pipeline is in. If it's not the pipeline won't be able to find the file. Another important thing is the formatting of the .csv file. There should be a single column of data with the first row of that column labeled "Address" and every following row will contain the URL. An example can be seen in the picture below.
PICTURE OF HOW THE CSV SHOULD LOOK

word_count_filter : This setting sets the minimum amount of words that are allowed in an article. For example, if the word count is set to 150 any article that contains less than 150 words will not be included in the data frame. 

repeated_phrase_filter : This setting is used to prevent any scrapping blockers from runining the data. For example, if a website puts up a blocker and the scrapper just returns "Access Denied" over and over again this setting prevents that article from being added to the dataframe.

social_starts_with : This setting is used to prevent social media sites from being scraped. These sites usually don't get scrapped very well and pertain to people's opinions rather than articles themselves. It's important to note that you can add to this list, and it doesn't pertain to just social media sites. If you notice that a specific site doesn’t scrape very well, you can always add it to this list.

url_max : This setting is used to set how many articles are scraped. If you attach a CSV with 3000 rows but only want to scrape 300 you can set this to 300 and scrape only those 300. If you don't want a limit you can set this to -1.

label_adjuster_on : This setting is used to create dynamic sentiment labels. This allows for more accurate sentiment labeling. For example, if you have articles about a war, most all articles are going to be labeled negative because of words like 'death', 'war', 'kill', etc. Turning this setting on creates labels based off the sentiment percentiles, for example every article that has a sentiment in the 40-60 percentile is neutral while every article in the 80-100 percentile is labeled positive. 

topic_model_dict : This setting contains the settings for topic modeling. topic_limit is the number of topics that the model will find within the space. topic_start is the number of topics we start at and topic_step is how many we increment every step.

## Pipeline Start
This is where the webscrapping, sentiment analysis, and the topic modeling begin. The first block is the Sentiment Analysis Pipeline. Within this block the articles are scraped, then they are run through sentiment analysis, and a data frame is returned.

Then the topic modeling starts. The function will create topic models with varying number of topics, compute the topic coherence score for that number of topics over our articles, and return to us the LDA model that has the highest coherence score. It will display a small graphic showing the topic coherence for each number of topics we tested. Next, we will create our topic-level sentiment analysis dictionary using both our dataframe df and the new LDA_model and corpus we just created. This will give us a dictionary of the average sentiment scores for every topic of an article, for all articles.

After this section is ran all rows and columns have been added to the data frame.


## Data Visualization Setup
The first code block is a preprocessing block to get our data frame in the proper format for data visualizations.

The second block is the settings block for the data visualizations. There are two settings:

list_of_topics_to_visualize : This setting is used to customize which topics will be displayed individually.

kmeans_settings : max_clusters is used in the inertia visualization (shown below) and num_clusters is used to define how many clusters will be shown in the k-means visualization. pca_components should be left at the default of 2.

## Data Visualization
From this point on all the visualizations are displayed. All visualizations will include a picture of the sample output. Here is the settings that were applied when the sample output was generated:
SETTINGS HERE

### Display the main data frame
This section shows the data frame.


### Display the topic model words
This section shows the words associated with each topic.


### Visualize all articles on their main topic
Here we will show two simple graphs plotting all articles' sentiment values, with the articles sorted by main topic. This is to show the distribution of sentiment towards the different topics. The scatter plot will let you look at each article individually, to see how each article contributes to the sentiment distribution. The box plot will show you the actual sentiment distribution for each topic and give details on the variance and average sentiment of a topic.

### Generating 2D and 3D Cluster Graphs of Topics (T-SNE)
This visualization generates a 2D clustering graph representing each article as a point in the topic space, which uses t-SNE dimensionality reduction and clustering. The goal of this visualization is to show every article compared to each other by their similarity in topics. You should see clusters of articles that share a lot of topic words and overall themes, while articles that differ in their topics are shown far apart. Colored by an article's main topic.


This generates a 3D clustering graph where the x and y-axis represent each article in the topic space, and the z-axis shows the sentiment value for each article. The 2D xy-plane is exactly the same as the 2D graph generated above, but we add a 3rd dimension in the form of sentiment to show how article clusters differ by their sentiment value as well as their topics. Colored by main topic.


### Subjectivity vs Sentiment of Articles for a single topic
Will iterate through our list_of_topics_to_visualize and will make a graph for each topic number in that list as well as that topic's word cloud of the top 10 topic words for that topic. So, for each topic we want to look at, it will show all the articles with that topic as their main topic, and plot the articles on their subjectivity vs sentiment. A higher subjectivity will show you that the author chose more opinionated words, rather than relying on neutral, factual statements. Compare that with the sentiment value to show whether they are positive and negative towards that topic, and how factual vs. opinionated their claims were. The word cloud will simply tell you what words are in that topic.


### Generating 2D and 3D Cluster Graphs of Topics (K means Clustering)
Generate a visual of number of topic clusters vs inertia (WCSS score) of the model (low inertia is good). We want to find the optimal number of topic clusters to use, and you can use the "elbow method" on this graph to find it. Ideally, the number of clusters is as small as possible, and the inertia is also as small as possible. So, to find the point where increasing the number of clusters gives diminishing returns on the inertia score, we imagine the line graph as an arm and use the number of clusters at the "elbow", or the point on the graph where the slope becomes much closer to 0 than the point before.

This will generate the topic space in 2D with every article represented as a single point. Very similar to the t-SNE clustering graph, this visualization has the same goal: show topic clusters of articles that are very similar in topic and overall theme. Points closer together means they have very similar topics, while points further apart will be less related. This visualization uses the relevance of each topic to an article as the data behind the clustering. Articles are colored by the user-selected number of clusters (not by main topic).

This visualization is a 3D version of the one above, with a 3rd dimension added to represent sentiment value of each article. So, the 2D xy-plane will be exactly the same as the visualization above--meant to represent articles in the topic space and how they are related by topic. The third dimension will show every article's sentiment value, to show how clusters or individual articles differ by sentiment (or are closely related!)


### Relevance X Sentiment of Articles K-Means
This visualization is a 2D clustering (using k-means) of the sentiment-weighted topic relevance values of every article. By that, we mean we have the relevance values of each article for every topic. We multiply each of those relevance values by that topic's associated sentiment value for an article, giving us a weighted relevance value based on the sentiment of that article towards that topic.

With that information, we use the same clustering algorithm as the 2D k-means graph above to produce a 2D topic space for every article, but this time it is weighted by the article's sentiment towards the topics.


### Sentiment vs Relevance Per Topic


### Topic Space

### Document Sentiment in Topic Space

### Document Density in Topic Space

### Topic per Document in Topic Space

### Sentiment Sum Heatmap in Topic Space


## References
Anderson, Martin. “MIT: Measuring Media Bias in Major News Outlets with Machine Learning.” Unite.AI. Unite.AI, December 10, 2022. https://www.unite.ai/mit-measuring-media-bias-in-major-news-outlets-with-machine-learning/.  
“Army Cyber Institute (ACI) Home.” Army Cyber Institute (ACI) Home. Accessed April 20, 2023. https://cyber.army.mil/. 
“Balanced News via Media Bias Ratings for an Unbiased News Perspective.” AllSides. AllSides, October 20, 2022. https://www.allsides.com/unbiased-balanced-news.
Chen, Wei-Fan, Benno Stein, Khalid Al-Khatib, and Henning Wachsmuth. “Detecting Media Bias in News Articles Using Gaussian Bias ... - Arxiv.” Detecting Media Bias in News Articles using Gaussian Bias Distributions. Accessed April 20, 2023.  https://arxiv.org/pdf/2010.10649.pdf. 
“Gensim: Topic Modelling for Humans.” Radim ÅehÅ¯Åek: Machine learning consulting, December 21, 2022. https://radimrehurek.com/gensim/. 
“Let's Build from Here.” GitHub. Accessed April 20, 2023. https://github.com/. 
“Media Bias/Fact Check News.” Media Bias/Fact Check, July 22, 2021. https://mediabiasfactcheck.com/. 
“Pandas.” pandas. Accessed April 20, 2023. https://pandas.pydata.org/. 
Pritchard, Jonathan K, Matthew Stephens, and Peter Donnelly. 2000. “Inference of Population Structure Using Multilocus Genotype Data.” Genetics 155 (2): 945–59. https://doi.org/10.1093/genetics/155.2.945.
Samantha D'Alonzo and Max Tegmark, “Machine-Learning Media Bias,” arXiv.org (Dept. of Physics and Institute for AI &amp; Fundamental Interactions, August 31, 2021), https://arxiv.org/abs/2109.00024.
“Spacy · Industrial-Strength Natural Language Processing in Python.” · Industrial-strength Natural Language Processing in Python. Accessed April 20, 2023. https://spacy.io/. 
“Simplified Text Processing.” TextBlob. Accessed April 20, 2023. https://textblob.readthedocs.io/en/dev/. 
Zhang, Xinliang. 2022. “Launchnlp/BASIL.” GitHub. July 17, 2022. https://github.com/launchnlp/BASIL. 

