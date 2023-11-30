# Probabilistic News Model
## Created by the Probabilistic News Team Gabriel Sams, William Hiatt, Deven Biehler, and Professor Balasubramanian ‘Subu’ Kandaswamy
Sponsoring Organization: The Army Cyber Institute (ACI) at West Point is our sponsoring organization. ACI is responsible for performing advisement, research, and analysis on any and all things within the cyberspace domain in support of the Army’s strategic objectives. 
Sponsor: Our sponsor and contact is MAJ Iain Cruickshank.

The ACI wanted to develop a non-deterministic model that allows social scientists and other domain experts to gain insight into a news article space. The goal of this project is to combat the propagation of misinformation by identifying bias and establishing connections among news articles based on their biases. 
The project should take news articles and provide the major topics of every article as well as the articles’ opinions/attitudes on the topics they discuss. The model should provide meaningful visualizations that give insight into how articles are related by opinion and topics. The Probabilistic News Model project addresses this problem by creating a new machine-learning pipeline through a combination of research and code exploration, ultimately creating a customizable application that can handle varying sizes of news article databases. 
This project was implemented using Python and associated Python libraries It uses a machine learning pipeline architecture with clustering methods and visualization. With our project, ACI at West Point now has a method of measuring news article sentiment and topics without human bias, and can rapidly assess a large number of articles without humans having to read and annotate them. The project is also designed to support new additions to the code and swapping out Python libraries for contemporary libraries so that it can be expanded upon in the future.


## How to use the Model
The model is broken into several different .ipynb files. To do any type of analysis only the Project_Pipeline.ipynb folder needs to be used. If you aren't making any code changes and just doing analysis there is no need to go into the other files. Therefore, this readme is going to go into depth on only the pipeline.

### Installs
**Run on initial startup.**
The very first code block of Project_Pipeline.ipynb includes all the installations one should need to use the notebook. You can run the entire code block and it will auto-install for you. You can see this code block below:
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
**Run on every startup.**

After our installs code block, we have our imports. If all the installs are installed successfully this code block should run without issue. This code block is used to import all the dependencies for the model.

### Connect Helper Files
**Run on every startup.**

This code block is used to connect all the helper files to the main pipeline. This code block allows for the pipeline to be condensed and easy to read/use. The files are nicely labeled so it's easy to see which file does what.
```
%run SentimentAnalysis.ipynb
%run WebScraper.ipynb
%run PipelineHelpers.ipynb
%run TopicModeling.ipynb
%run DataVisualization.ipynb
```



## Side Note
The sections from this point down will need to be run every time you want to analyze a new set of articles or change settings. Installs only need to be run the first time you use this project. Imports and Connect Helper Files need to be run when you first open the file but do not need to be run every time you try a new set of articles or change settings.

### General Pipeline Settings, and Pre-Analysis Setup
This code block is where you can change the settings of the web scrapping and topic analysis. In addition, this is also where you put the URL to the .csv file you want scraped. We will walk through each setting in this section.

`csv_file` : This setting is the actual .csv file to be scrapped. It is important that this file is stored within the same folder as the pipeline is in. If it's not the pipeline won't be able to find the file. Another important thing is the formatting of the .csv file. There should be a single column of data with the first row of that column labeled "Address" and every following row will contain the URL. An example can be seen in the picture below.

<img width="800" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/8a1d7e1c-9792-42ad-8072-1e9439fbaf44">

`word_count_filter` : This setting sets the minimum amount of words that are allowed in an article. For example, if the word count is set to 150 any article that contains less than 150 words will not be included in the data frame. 

`repeated_phrase_filter` : This setting is used to prevent any scrapping blockers from running the data. For example, if a website puts up a blocker and the scrapper just returns "Access Denied" over and over again this setting prevents that article from being added to the data frame.

`social_starts_with` : This setting is used to prevent social media sites from being scraped. These sites usually don't get scrapped very well and pertain to people's opinions rather than the articles themselves. It's important to note that you can add to this list, and it doesn't pertain to just social media sites. If you notice that a specific site doesn’t scrape very well, you can always add it to this list.

`url_max` : This setting is used to set how many articles are scraped. If you attach a CSV with 3000 rows but only want to scrape 300 you can set this to 300 and scrape only those 300. If you don't want a limit you can set this to -1.

`label_adjuster_on` : This setting is used to create dynamic sentiment labels. This allows for more accurate sentiment labeling. For example, if you have articles about war, almost all articles are going to be labeled negative because of words like 'death', 'war', 'kill', etc. Turning this setting on creates labels based on the sentiment percentiles, for example, every article that has a sentiment in the 40-60 percentile is neutral while every article in the 80-100 percentile is labeled positive. 

`topic_model_dict` : This setting contains the settings for topic modeling. `topic_limit` is the number of topics that the model will find within the space. `topic_start` is the number of topics we start at and `topic_step` is how many we increment every step.



## Pipeline Start
This is where the web scrapping, sentiment analysis, and topic modeling begin. The first block is the Sentiment Analysis Pipeline. Within this block the articles are scraped, then they are run through sentiment analysis, and a data frame is returned.

Then the topic modeling starts. The function will create topic models with varying numbers of topics, compute the topic coherence score for that number of topics over our articles, and return to us the LDA model that has the highest coherence score. It will display a small graphic showing the topic coherence for each number of topics we tested. Next, we will create our topic-level sentiment analysis dictionary using both our data frame df and the new `LDA_model` and `corpus` we just created. This will give us a dictionary of the average sentiment scores for every topic of an article, for all articles.

After this section is run all rows and columns have been added to the data frame.



## Data Visualization Setup
The first code block is a preprocessing block to get our data frame in the proper format for data visualizations.

The second block is the settings block for the data visualizations. There are two settings:

`list_of_topics_to_visualize` : This setting is used to customize which topics will be displayed individually.

`kmeans_settings` : `max_clusters` is used in the inertia visualization (shown below) and `num_clusters` is used to define how many clusters will be shown in the k-means visualization. `pca_components` should be left at the default of 2.



## Data Visualization
From this point on all the visualizations are displayed. All visualizations will include a picture of the sample output. Here is the settings that were applied when the sample output was generated:
SETTINGS HERE

### Display the main data frame
This section shows a snippet of the data frame.
<img width="1637" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/0887f3ff-ac84-4cc8-a97f-e16573134f3e">

### Display the topic model words
This section shows the words associated with each topic.

<img width="100" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/bab9ff59-57e1-4ecc-af22-98a719498cd7">


### Visualize all articles on their main topic
Here we will show two simple graphs plotting all articles' sentiment values, with the articles sorted by main topic. This is to show the distribution of sentiment towards the different topics. The scatter plot will let you look at each article individually, to see how each article contributes to the sentiment distribution. The box plot will show you the actual sentiment distribution for each topic and give details on the variance and average sentiment of a topic.

<img width="1571" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/12da282b-1ded-4384-8473-459d7f70d655">

<img width="1571" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/e4acc0ec-480e-412f-b22b-859f96d6c1b8">



### Generating 2D and 3D Cluster Graphs of Topics (T-SNE)
This visualization generates a 2D clustering graph representing each article as a point in the topic space, which uses t-SNE dimensionality reduction and clustering. The goal of this visualization is to show every article compared to each other by their similarity in topics. You should see clusters of articles that share a lot of topic words and overall themes, while articles that differ in their topics are shown far apart. Colored by an article's main topic.

<img width="637" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/eb00146b-5824-4692-9dc8-84a9e99f4d43">

This generates a 3D clustering graph where the x and y-axis represent each article in the topic space, and the z-axis shows the sentiment value for each article. The 2D xy-plane is exactly the same as the 2D graph generated above, but we add a 3rd dimension in the form of sentiment to show how article clusters differ by their sentiment value as well as their topics. Colored by main topic.

<img width="626" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/5b3cea1e-0c29-48f9-9d1a-314cbd3aa8c3">


### Subjectivity vs Sentiment of Articles for a single topic
Will iterate through our `list_of_topics_to_visualize` and will make a graph for each topic number in that list as well as that topic's word cloud of the top 10 topic words for that topic. So, for each topic we want to look at, it will show all the articles with that topic as their main topic, and plot the articles on their subjectivity vs sentiment. A higher subjectivity will show you that the author chose more opinionated words, rather than relying on neutral, factual statements. Compare that with the sentiment value to show whether they are positive and negative towards that topic, and how factual vs. opinionated their claims were. The word cloud will simply tell you what words are in that topic.

<img width="610" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/b2643c61-12fc-45fc-85a1-921e5f991925">

<img width="314" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/6a238f16-16ce-44aa-954f-9c6b38ded335">


### Generating 2D and 3D Cluster Graphs of Topics (K means Clustering)
Generate a visual of the number of topic clusters vs inertia (WCSS score) of the model (low inertia is good). We want to find the optimal number of topic clusters to use, and you can use the "elbow method" on this graph to find it. Ideally, the number of clusters is as small as possible, and the inertia is also as small as possible. So, to find the point where increasing the number of clusters gives diminishing returns on the inertia score, we imagine the line graph as an arm and use the number of clusters at the "elbow", or the point on the graph where the slope becomes much closer to 0 than the point before.

<img width="458" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/6930ce02-b310-4955-9095-a8fb44b0049d">


This will generate the topic space in 2D with every article represented as a single point. Very similar to the t-SNE clustering graph, this visualization has the same goal: show topic clusters of articles that are very similar in topic and overall theme. Points closer together mean they have very similar topics, while points further apart will be less related. This visualization uses the relevance of each topic to an article as the data behind the clustering. Articles are colored by the user-selected number of clusters (not by the main topic).

<img width="1608" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/921999f5-31b5-40ef-a3a3-c29b6eb75389">

This visualization is a 3D version of the one above, with a 3rd dimension added to represent the sentiment value of each article. So, the 2D xy-plane will be exactly the same as the visualization above--meant to represent articles in the topic space and how they are related by topic. The third dimension will show every article's sentiment value, to show how clusters or individual articles differ by sentiment (or are closely related!)

<img width="308" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/9dd19680-d334-4fdb-8da8-274efd3c3b23">


### Relevance X Sentiment of Articles K-Means
This visualization is a 2D clustering (using k-means) of the sentiment-weighted topic relevance values of every article. By that, we mean we have the relevance values of each article for every topic. We multiply each of those relevance values by that topic's associated sentiment value for an article, giving us a weighted relevance value based on the sentiment of that article towards that topic.

With that information, we use the same clustering algorithm as the 2D k-means graph above to produce a 2D topic space for every article, but this time it is weighted by the article's sentiment towards the topics.

<img width="1616" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/6bc2ef17-c505-4715-b380-c95e2908241b">


### Sentiment vs Relevance Per Topic
A scatter plot of sentiment values against relevance values of each selected topic. This will display a scatter plot with sentiment values on the x-axis and relevance values on the y-axis, along with a tooltip.

<img width="452" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/379ff5d6-1a99-49f8-b299-387e8ee641e1">


### Topic Space
Applying principle component analysis(PCA) will reduce the high-dimensional topic space into a simple two-dimensional space. 2 dimensions are easy for us to interpret but also hold a lot of information.

Visualize the topic space after principle component analysis(PCA) is applied to reduce the topic model results to two dimensions. Plots each point in the topic space, the distance is directly related to how similar the topics are to each other. This will expose any outlier topics as well as topics that might be quite similar. The ideal plot is one where topics are equal distance from each other.

<img width="376" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/48753580-b1c6-49a3-a237-505e073ce119">


### Document Sentiment in Topic Space
Here is where each document is added to the topic space, the documents are closer or further from each topic based on the relevance score given to each document. Then a color is added to each document to represent the sentiment score. Red represents a negative sentiment while green represents a positive sentiment.

<img width="292" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/3036e163-cdf4-49e3-9623-c7bc4b4019d4">


### Document Density in Topic Space
This plots the density of documents that were plotted in the topic space. Red is hot zones where many documents fall while white is where very few documents show up. This will help us understand the topics that the majority of the articles are covering.

<img width="448" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/65a88149-7e0a-49e0-9b37-7677c4ebdcc0">


### Sentiment Sum Heatmap in Topic Space
This graph plots the normalized sentiment in each area of the topic space. Green is the more positive topics, while red is more negative. Each document's sentiment is added to the area of the heatmap and then normalized to be able to visualize where the positive and negative topics lie in the topic space more easily.

<img width="308" alt="image" src="https://github.com/WSUCapstoneS2023/Media-Bias-Prediction-Model/assets/98793239/83cca970-5d08-42d0-885d-e17035d09503">


## References
Anderson, Martin. “MIT: Measuring Media Bias in Major News Outlets with Machine Learning.” Unite.AI. Unite.AI, December 10, 2022. https://www.unite.ai/mit-measuring-media-bias-in-major-news-outlets-with-machine-learning/

“Army Cyber Institute (ACI) Home.” Army Cyber Institute (ACI) Home. Accessed April 20, 2023. https://cyber.army.mil/

“Balanced News via Media Bias Ratings for an Unbiased News Perspective.” AllSides. AllSides, October 20, 2022. https://www.allsides.com/unbiased-balanced-news

Chen, Wei-Fan, Benno Stein, Khalid Al-Khatib, and Henning Wachsmuth. “Detecting Media Bias in News Articles Using Gaussian Bias ... - Arxiv.” Detecting Media Bias in News Articles using Gaussian Bias Distributions. Accessed April 20, 2023.  https://arxiv.org/pdf/2010.10649.pdf

“Gensim: Topic Modelling for Humans.” Radim ÅehÅ¯Åek: Machine learning consulting, December 21, 2022. https://radimrehurek.com/gensim/

“Let's Build from Here.” GitHub. Accessed April 20, 2023. https://github.com/

“Media Bias/Fact Check News.” Media Bias/Fact Check, July 22, 2021. https://mediabiasfactcheck.com/

“Pandas.” pandas. Accessed April 20, 2023. https://pandas.pydata.org/

Pritchard, Jonathan K, Matthew Stephens, and Peter Donnelly. 2000. “Inference of Population Structure Using Multilocus Genotype Data.” Genetics 155 (2): 945–59. https://doi.org/10.1093/genetics/155.2.945

Samantha D'Alonzo and Max Tegmark, “Machine-Learning Media Bias,” arXiv.org (Dept. of Physics and Institute for AI &amp; Fundamental Interactions, August 31, 2021), https://arxiv.org/abs/2109.00024

“Spacy · Industrial-Strength Natural Language Processing in Python.” · Industrial-strength Natural Language Processing in Python. Accessed April 20, 2023. https://spacy.io/

“Simplified Text Processing.” TextBlob. Accessed April 20, 2023. https://textblob.readthedocs.io/en/dev/

Zhang, Xinliang. 2022. “Launchnlp/BASIL.” GitHub. July 17, 2022. https://github.com/launchnlp/BASIL


