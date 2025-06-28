# Summary 🚀
The main task of this competition is to detect fake texts in a given dataset. Each data sample contains two texts - one real and one fake. The data used in this Challenge come from The Messenger journal. Importantly, both texts in each sample - real (optimal for the recipient, as close as possible to the hidden original text) and fake (more or much more distant from the hidden original text) - have been significantly modified using LLMs.

You have access to:
1) a specially curated dataset designed for this competition,
2) Jupyter notebooks with a baseline solution and a sample submission.

The Impostor Hunt starts now! 🚀

Description 📜
In 2021, ESA's European Space Operations Centre initiated a project to outline a clear direction for AI implementation in mission operations. This initiative led to the creation of the Artificial Intelligence for Automation (A2I) Roadmap [1], developed by a team of over 70 specialists at ESOC, with support from an industry consortium. One of the key elements of the Roadmap is the security of AI applications addressed within the Assurance for Space Domain AI Applications project and its Catalogue of Security Risks for AI Applications in Space. This competition is based on two real-life AI security threats identified within the project - data poisoning and overreliance.

Story
A company uses LLM(s) to perform operations on its official document (e.g., summarization). Different models are used for this purpose and the history of which model with which setting performed which operation on which document is not strictly tracked.

In the meantime, it was found that sometimes the models malfunction and may provide harmful output (hallucination, hidden triggers changing the facts in the documents). The reason for this behaviour has yet to be discovered, but for now, there is a more critical task - detection, which text is correct (real) and which was changed (fake).

Fortunately, a few days ago, one of your colleagues found out that there is a small set of texts for which there are both real and fake texts available, and for some of them, someone managed to detect which one is real and which is fake. Unfortunately, there is no way to continue the task since the person has gone missing and no code was found.

You are the company's data scientist, who was asked to continue this job and create the algorithm to distinguish between real and fake documents.

Task description
You are provided with a collection of documents. Some of them are real (optimal for the recipient, as close as possible to the hidden original text), and some of them are fake (more or much more distant from the hidden original text). The documents are divided into pairs, one of which is always real and one fake, but there is no specific order in which one is which in a given pair. They were sorted randomly.

The main topic of the documents is research and organization in space-related projects. They may contain information about scientific devices, research outcomes, space-related workshops or interesting people in engineering, science, or even astronauts. The documents are public, but it is not necessary to have online access to complete tasks. Each of the documents has been significantly modified using LLMs. The language of all original documents is English.

Summarizations were made using a standard prompt that is not provided but is not necessary to complete tasks. Note that even for decent models, the summarization may not be satisfying in each case. Tasks are created to distinguish between mostly always good outputs and corrupted outputs, so this should not be an issue.

The texts have been modified in several different ways. Be aware that not all modifications may be provided in public test sets. Therefore, manual solutions or rule-based solutions might not be the best idea.

Data
More details regarding the data are provided in the Data tab.

Code
The competition offers a notebook to reproduce the results of baseline algorithms directly at Kaggle and show sample solution format.

Prizes
Monetary prizes for the top 3 teams:
-🥇500 USD
-🥈250 USD
-🥉150 USD

The award ceremony and best teams presentations will take place during the next ESA AI STAR conference (date TBA). We are also in the process of organizing a workshop about security of AI at a top machine learning conference. Winner(s) will be invited as co-authors of a joint paper summarizing the competition. Thus, sharing the details of the solution will be necessary to be eligible for the final prize.

Please check the Rules tab for further details about eligibility and criteria.

Hackathon
The Challenge will be followed by a hackathon (more details and date TBA).

Evaluation metric ✅
Submissions are evaluated using the pairwise accuracy, which evaluates how well are the models able to align with human preferences, which text should be preferred in a given pair.

In summary, the proposed solutions should focus on minimizing the number of wrongly chosen texts from each of the pairs.
The metric does NOT care about computation timing, however please have in mind the Kaggle execution time limits: 12 hours for CPU and GPU notebook sessions and 9 hours for TPU notebook sessions.

Submission file 📁
For each index representing a consecutive pair of texts in the test set, the solution file should contain a number indicating, which of the texts is real. The file should contain a header and have the following format:

id	real_text_id
1501	1
1502	1
1503	2
1504	1
…	…
The file should be saved in CSV.

Timeline ⌛
Start Date - 23 June 2025
Entry deadline - Same as the Final Submission Deadline
Team Merger deadline - Same as the Final Submission Deadline
Final submission deadline - 23 September 2025
Private leaderboard release - 30 September 2025
All deadlines are at 12:00 PM CEST on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if deemed necessary.

Organizers Team ✨
The competition is organized on behalf of ESA (Evridiki Ntagiou) by MI Space Team from Warsaw University of Technology (Przemysław Biecek, Agata Kaczmarek and Dawid Płudowski) and KPLabs (Krzysztof Kotowski and Ramez Shendy).

References 📖
[1] G. De Canio, J. Eggleston, J. Fauste, A. M. Palowski, and M. Spada, “Development of an actionable AI roadmap for automating mission operations,” in 2023 SpaceOps Conference, Dubai, United Arab Emirates, link

Citation
Agata Kaczmarek, Dawid Płudowski, Evridiki Ntagiou, and Krzysztof Kotowski. Fake or Real: The Impostor Hunt in Texts. https://kaggle.com/competitions/fake-or-real-the-impostor-hunt, 2025. Kaggle.

# Data
Dataset Description
TLDR
The data used in this Challenge come from many different The Messenger journals. Importantly, both texts in each sample (fake and real) have been significantly modified using LLMs.

Your task is to find which text in each pair is considered Real.

List of directories and files
train - directory
test - directory
train.csv - file with ground truth for samples in train directory
The data in the folder are divided into train and test directory. There is also file train.csv with ground truth for each sample from train directory.

In both folders, train and test there are different articles, randomly numbered. For each of such articles, there are two versions provided file_1.txt and file_2.txt, one is treated as Real and one is Fake, they are also randomly placed as number 1 and number 2.

In train.csv file there is ground truth provided, first column is an id of an article, second is real_text_id, so which of the files consists of Real sample (number 1 or 2). Sample format of submission is shown in Overview tab.

More details
Preprocessing
The dataset is a highly preprocessed version of the raw The Messenger journal articles (both
Real and Fake). For the preprocessing, different techniques were used. LLMs were also used.