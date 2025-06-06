# Email-phishing-detection
In the present repository are all the data for testing / running our solution.
In each folder, in the file "index", it is a brief description of the items present
in that specific folder

First thing it is to train the Llama model, which will be automatically saved on your local storage device.
We used the model available on https://huggingface.co/meta-llama (Llama-3.2-3B-Instruct). The script will 
automatically retrive the model. In order to do this you must have a valid account on Hugging Face portal.
 
In this repository we have three scripts (in the dolfer "Scripts") that performs email phishind detection, as follows :
* Llama - where we use only the Llama model ( the trained model) to identify and to report the nature of an Email as
Safe or Phishing;
* Bayes Classifiers - a classical approach without the use of LLM's;
* Hybrid solution - using trained Llama model, a set of trigger patterns and Bayes Classifiers.


"Libraries used in the project" - it is a list with all the libraries used in our script;

In the presented folders : 
    "datasets" - are the datasets used for testing the Llama solution and hybrid solution.
                 For testing the Llama solution the dataset it is : test.csv
                 For testing the hybrid solution the dataset it is : training_emails_big.csv
    "tokens" - are the tokens lists used in our solution.
                Stop words - the stop words that we will not take into consideration
                Token categories - the categories defined by us based upon the most frequent used words
                Trigger patterns - the list with the trigger patterns that will command the function of the script
    "scripts" - here are the scripts that we have used
               fine_tune_model.py - for training the META-Llama model for email phishing detection
               Llama_solution.py - for testing the performance of the META-Llama model in identifing the nature of an email:
               Safe Email / Phishing Email
               Bayes_solution.py - for testing the performance of Bayes classifiers in identifing the nature of an email:
               Safe Mail / Phishing Email
               Hybrid_solution.py - for testing the hybrid solution , proposed by us : running Llama, trigger phrasses and Bayes
               Classifiers
    "results" - in this folders we have put the results that we have obtained running the scripts. We put it here for 
                banchmark / evaluation. Below you will find also a brief description of the system on which we have tested
                the solutions.
               
   
Other considerations : 
* code was written in Python (3.11) using PyCharm (IDE)
* we also have used Cuda for GPU
* system used : Asus Tuf Gaming F15 laptop, having the following configuration :
             ** Intel i7-13620H 13th Generation processor;
             ** 32 Gb of RAM in dual channel;
             ** dedicated video card nVidia GeForce RTX4060 with 8 Gb of RAM, DDR6;
             ** 2 nVme SSD : 1 Micron 500Gb for the operating system, 1 Samsung 990 PRO 2Tb for data;
             ** Windows 11 Pro,Version 24H2.

