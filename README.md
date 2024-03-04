# Project 2 Predictors and Classification of Consumer Decisions and Attitudes Toward Electric Vehicles

Team Members:  Ryan Busman, Elaine Kwan, Deb Peters, Nick Watkins

![Presentation_Cover_Image](<Presentation/Images/Project 2 Cover Page.png>)

##Purpose
---
At the conclusion of Project 1, it was apparent to team members that consumers' beliefs and attitudes about electric vehicles (EVs) could have a substantial impact on purchase or intent to purchase an EV.  However, the exploratory data analyses were limited to demographic data that did not probe deeper into consumer attitudes that could influence the decision to purchase an EV. 

For Project 2, we opted to take a deeper dive into the consumer sentiments' that may impact intent to purchase an EV.  

After isolating key drivers of intent to purchase an EV, we aimed to find an optimal classification model that could accurately predict the future of EVs and the energy sector. 


##Audience: 
----

Our project is timely.  Since Project 1 and the beginning of 2024, EV sales have plummeted and car manufacturers have scaled back on their production of EVs.  The car industry is eager to understand what may be driving this shift and how to address it.  Audience members for our Project 2 research include:
- Policymakers 
- Vehicle Manufacturers 
- EV Engineers 
- EV sales and marketing 
- EV consumers/individual consumers 

##About the Data:
----
Pew Research Center: Wave 108 of the American Trends Panel: COVID
and Climate, Energy and the Environment
▪ https://www.pewresearch.org/american-trends-panel-datasets/

• "The ATP is Pew Research Center’s nationally representative
online survey panel. The panel is composed of more than
10,000 adults selected at random from across the entire U.S."

• "For the online panel to be truly nationally representative, the
share of those who do not use the internet must be
represented on the panel somehow. In 2021, the share of
non-internet users in the U.S was estimated to be 7%, and
while this is a relatively small group, its members are quite
different demographically from those who go online. In its
early years, the ATP conducted interviews with non-internet
users via paper questionnaires. However, in 2016, the
Center switched to providing non-internet households with
tablets which they could use to take the surveys online. The
Center works with Ipsos, an international market and opinion
research organization, to recruit panelists, manage the panel
and conduct the surveys."

• "We (Pew Research Center) make(s) a promise to our panelists to protect their identity.Several checks and balances are in place to make sure that Pew Research Center remains true to its word. Personal identifying information (PII) such as a panelist’s name or county of residence is maintained solely by the core panel administration team and is never made available to the general public. In some cases, additional steps such as data
swapping – randomly swapping certain values among a
small number of respondents with similar characteristics for sensitive questions – is also used to protect panelists’
information."

> **Licensure and Credits:**
> - Open access
> - "When using Pew Research Center data, ensure proper attribution by mentioning the source. For example: 'Data from the Pew Research Center, [dataset title], [date of data collection], [URL to dataset].'"
> - "Include the specific dataset title and the date of data collection."

#Data Implementation
----
##Data Implementation: Data Cleaning and Preparation
Data Cleaning and Preparation is necessary so that we can access our data and implement a functional dataframe for our coding during later stages of Data Implementation and during Data Model Evaluation and Optimization.

This stage of Data Implementation includes feature selection from the ATP questionnaire, detecting and erasing null values, detecting and erasing duplicate entires, and recoding features so that results from analyses are interpretable. A Data Key was generated during this phase to document feature and feature characteristics and guide later coding and interpretation.

This version of the dataframe does not have EVCAR2B (i.e., "How much of a reason is each of the following for why you would consider purchasing and electric vehicle?") as only respondents who indicated they were very likely to purchase an EV were asked EVCAR2B, thereby greatly reducing our sample size.

PLACEHOLDER FOR BLOCKS OF CODE WITH EXPLANATIONS...




# Directory Structure
---
- Project-2
contains subfolders and files
- About_Pew_ATP/
contains methodology topline and codebook_markedup
- Code/
contains all code by topic
- Data/
contains all datasets by topic
- Presentation/
contains the final presentation and its images
- Project2_Proposal_EVs
contains initial proposal submission
- README.md
contains project details and definitions