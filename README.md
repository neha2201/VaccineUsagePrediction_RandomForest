# VaccineUsagePrediction_RandomForest

## **Problem Statement**

  Predict how likely it is that the people will take an H1N1 flu vaccine using Random Forest.
  
## **Purpose**

   In spring 2009, an epidemic caused by the H1N1 influenza virus, also commonly known as "swine flu," swept across the world. Researchers estimate that it was responsible for        between 151,000 to 575,000 deaths globally in the first year.
   
   The U.S. government conducted the National 2009 H1N1 Flu Survey (NHFS) via phone. The survey asked respondents whether they had received the H1N1 and seasonal flu vaccines with    questions about themselves. These additional questions covered respondents' various information such as background (social, economic, and demographic), opinions on risks of        disease and vaccine effectiveness, and so on. A better understanding of these characteristics and individual vaccination patterns can guide future public health efforts.
   
   
## **Roadmap**

  The result of a proper modelling totally depends on the quality of data as wel as uasage of proper parameters. Given below is the following steps for building a model.
  
  1. Business Understanding
  2. Data Collections
  3. Exploratory Data Analysis
  4. Cleaning of Data
  5. Feature Enginering
  6. Model Building
  7. Training|Validating Model
  8. Results and Interpretations
  
  ### **Business Understanding :**
  
  - This step involves understanding of business rules and terminology very well. It is the first and most important step of building a model. Without having proper knowledge         about business, one cannot make a proper model.
    
  ### **Data Collections :**
  
  - This step involves gathering of data, loading the data into the system. In this project I have downloaded the data from the kaggle platform. Below is the link given.
   
  ```javascript
          
           https://www.kaggle.com/c/prediction-of-h1n1-vaccination/data

  ```
  
  - Loading Dataset: The dataset consists of 26707 rows and 34 Columns.
  
  Table-1: List of variable with description available in Raw Dataset
  
  |       Columns                 |                              Descriptions                                                                           |
  |-------------------------------|---------------------------------------------------------------------------------------------------------------------|
  |      unique_id                |                                       Unique identifier for each respondent                                         |
  |      h1n1_worry               |      Worry about the h1n1 flu(0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried |
  |           h1n1_awareness      |                  h1n1 flu - (0,1,2) - 0=No knowledge, 1=little knowledge, 2=good knowledge                          |
  |       antiviral_medication    |                              Has the respondent taken antiviral vaccination - (0,1)                                 |
  |         contact_avoidance     |     Has avoided any close contact with people who have flu-like symptoms - (0,1)                                    |
  |    bought_face_mask           |      Has the respondent bought mask                                                                                 |
  |      wash_hands_frequently    |       Washes hands frequently or uses hand sanitizer -(0,1)                                                         |
  |   avoid_large_gatherings      |        Has the respondent reduced time spent at large gatherings - (0,1)                                            |
  |    reduced_outside_home_cont  |     Has the respondent reduced contact with people outside their own house - (0,1)                                  |      
  |    avoid_touch_face           |         Avoids touching nose, eyes, mouth - (0,1)                                                                   |
  |   dr_recc_h1n1_vacc           |      Doctor has recommended h1n1 vaccine - (0,1)                                                                    |                           |    cont_child_undr_6_mnth     |      Has regular contact with child the age of 6 months -(0,1)                                                      |                      
  |   dr_recc_seasonal_vacc       |      The doctor has recommended seasonalflu vaccine -(0,1)                                                          |                           |    chronic_medic_condition    |     Has any chronic medical condition - (0,1)                                                                       |   
  |   is_health_worker            |     Is respondent a health worker - (0,1)                                                                           |
  |    has_health_insur           |      Does respondent have health insurance - (0,1)                                                                  |                           |   is_h1n1_vacc_effective      | 1=not effective at all, 2=not very effective,3=Doesn't know effective or not 4=somewhateffective, 5=highly effective|
  |   is_h1n1_risky               | 1=not very low risk, 2=somewhat low risk, 3=don’t know risky or not, 4=somewhat high risk,5=highly risky            |                           |   is_seas_vacc_effective      | 1=not effective at all,2=not very effective,3=Doesn't know effective or not, 4=somewhat effective,5=highly effective|
  |  age_bracket                  |      Age bracket of the respondent - (18 - 34 Years, 35 - 44 Years, 45 - 54 Years, 55 - 64 Years, 64+ Years)        |
  |  qualification                |  Qualification/education level of the respondent  -(<12 Years, 12 Years, College Graduate, Some College)            |
  |   race                        |     Respondent's race - (White, Black, Other or Multiple ,Hispanic)                                                 |
  |    sex                        |                                Respondent's sex - (Female, Male)                                                    |
  |   income_level                |   Annual income of the respondent as per the 2008- (<=75000−AbovePoverty,>75000−AbovePoverty,>75000, Below Poverty) |
  |    marital_status             |       Respondent's marital status - (Not Married, Married)                                                          |
  |   housing_status              |   Respondent's housing status - (Own, Rent)                                                                         |
  |   employment                  |      Respondent's employment status - (Not in Labor Force, Employed, Unemployed)                                    |
  |  census_msa                   |  Residence of the respondent with the MSA(Non-MSA, MSANot Principle, CityMSA-Principle city) - (Yes, no)           |
  |   no_of_adults                |     Number of adults in the respondent's house (0,1,2,3) -(Yes, no)                                                 |
  |    no_of_children             |     Number of children in the respondent's house(0,1,2,3) - (Yes, No)                                               |
  |    h1n1_vaccine               |    Dependent variable)Did the respondent received the h1n1 vaccine or not(1,0) - (Yes, No)                          |
  
  ### **Exploratory Data Analysis :**
  
  - This step involves various analysis on the given datasets. It is the process of investigating the dataset to discover patterns, and anomalies (outliers), and form hypotheses     based on our understanding of the dataset with the help of summary statistics and graphical representations.


  - Graphical Analysis:

    ![image](https://user-images.githubusercontent.com/79011767/135742544-ac29e865-c777-4325-9ea4-0c6e1f27693c.png)
    - From the above diagram, we acan clearly say that respondants having high qualification have more awreness regarding flu.


    ![image](https://user-images.githubusercontent.com/79011767/135742682-26ffd368-108e-4a55-bcd0-0c895d052ec5.png)

  
  ### **Cleaning of  Data :**
 
  - Missing Value Treatment: In the given dataset, we have null values in most of the features.
       To fill the missing value for numerical features, I have build a function.
 
  ```javascript
          
           def Missing_imputation(x):
           x = x.fillna(x.median())
              return x

          vac=vac.apply(lambda x: Missing_imputation(x))

   ```
  
  - Outlier Treatment: Since there are no outliers present in the given dataset, I have not performed any outlier treatment.
    
  
    

 
      
