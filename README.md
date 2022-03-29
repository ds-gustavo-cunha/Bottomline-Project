# **BUSINESS UNDERSTANDING**

## **What is the problem we will try to solve?**

The problem we are tackling is related to the overwhelming amount of information being produced by the modern society. Information per se are valuable but often its very challenging to spotlight the essential and important part out of it - the bottomline, so to say. This mental-filtering process can be very time consuming and also confusing sometimes.

With our technical solution we provide an automated service, which identifies the most relevant sentences within text. 

# **PROBLEM UNDERSTANDING**

We claim that people rather use their mental ressources efficiently than inefficiently. Furthermore we claim that the current process, in wich informations are getting consumet and digested, has an siginifcant potential of improvement. We see AI as relevant enabler for this potential and therefore as problem solver. 

# **PROJECT CYCLE**

![crisp-ds](img/crisp_dm.png)

## Step 01. Business Understanding:
Understand what is the problem we are trying to solve and what are the main concerns of this problem.

## Step 02. Data Understanding:
Understand what data we have available and plan how to use data science tools to solve the business problem.

## Step 03. Data Preparation:
Remove punctuation, mispellings and irrelevant text and prepare the data for model training


## Step 04. Modelling:
Use machine learning and deep learning technique to extract patterns on the data and make relevant predictions.

## Step 05. Evaluation:
Evaluate the results from modelling and check if the product so far is able to be deployed or need further improvements to be deployed.

## Step 06. Deploy:
Deploy the product of the respective project iteration so that user can use the product and give it a feedback.

## Restart the cycle:
Use the feedback from last project iteration and also user feedback to make improvements to the next project cycle.

# **BUSINESS SOLUTION**

**You can check the whole solution architecture for this project at the following image**

![deployment-architecture](img/deployment_architecture.png)

**You can check our first MVP for this project at the following image**

![app-overview](img/bottomline_project-app_overview.png)

# **CONCLUSIONS**

Based on the first user feedback we defined the following ideas for further improvments:
1. Improve or remove the sentiment analysis 
2. Highlight the sentences in the input box, wich are being used in the final summary box
3. Explain better BERT or remove it
4. In the upper part: bring the summarized sentences in a order according to the inputed text
5. Reducing the seize of the word cloud

# **LESSONS LEARNED**

**How to do an end-to-end NLP Data Science project.**

**How to create the whole data architecture for the app (and its API) on Google Cloud Platform.**

**Use Machine Learning and Deep Learning techniques to do text summarization.**

**Use Machine Learning and Deep Learning techniques to do sentiment analysis.**

**How to create a user friendly app and also take the UX into account when designing data apps.**

**Work in cycles creating MVPs, validating its solution on user feedbacks and improving the solution on the next project cycle1.**

**Working on presentation and storytelling so even people without data science background could understand our project and use our final product**

# **NEXT STEPS TO IMPROVE**

**Sentiment Analysis**: train the model on a larger dataset.

**Data cleaning**: test data cleaning step on a larger range of possible misleading text input.

**User input**: allow user to input PDF and DOCX files.

**Web scraping**: make the web scraping more robust to different website HTML codes.

**UX**: improve UX for mobile usage.
