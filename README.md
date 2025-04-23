# Email-phishing-detection
In the present repository are all the data for testing / running our solution.
In each folder, in the file index, it is a brief description of the items present
in that specific folder

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
               ***** - for training the META-Llama model for email phishing detection
               ***** - for testing the performance of the META-Llama model in identifing the nature of an email:
               Safe EMail / Phishing Email
               **** - for testing the performance of Bayes classifiers in identifing the nature of an email:
               Safe Mail / Phishing Email
               **** - for testing the hybrid solution , proposed by us : running Llama, trigger phrasses and Bayes
               Classifiers
               
    First thing it is to train the Llama model, which will be automatically saved on your local storage device.
    We recommned to have at least 17 Gb free space available for the model. We used the model available on 
    https://huggingface.co/meta-llama. The script will automatically retrive the model. In order to do this you must have
    a valid account on Hugging Face portal.
    In this repository we have three scripts that performs email phishind detection, as follows :
    * Llama - where we use only the Llama model ( the trained model) to identify and to report the nature of an Email as
    Safe or Phishing
    * Bayes Classifiers - a classical approach without the use of LLM's
    * hybrid solution - using trained Llama model, a set of trigger patterns and Bayes Classifiers. This solution also
    uses the already trained model of Llama.

Other considerations : 
# code was written in Python (3.11) using PyCharm (IDE)
# we also have used Cuda for GPU

