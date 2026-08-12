#!/usr/bin/env python
# coding: utf-8

# In[53]:

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[54]:


df = pd.read_csv("Employee_Attrition_Data.csv", encoding='unicode_escape')
df


# In[55]:


df = df.rename(columns = {"ï»¿Age": "Age"})


# In[56]:


df.head()


# In[57]:


df.info()


# In[58]:


df.shape


# In[59]:


df.describe().round(2)


# In[60]:


df.nunique()


# ## <span style="color:orange;">[Employee_Attrition_EDA_Report]</span>

# <b style="color:red;">Drop the constant columns from the dataset.</b>

# In[61]:


df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])
df


# In[62]:


df.shape


# <h2 style="color:orange;">Correlation Heatmap - Relationships b/w Features</h2>

# <b style="color:red;">Numeric Columns</b>

# In[63]:


numeric_df = df.select_dtypes(include=["int", "float"])
numeric_df.columns


# <b style="color:red;">Categorical Columns</b>

# In[64]:


df.select_dtypes(exclude=["int", "float"]).columns


# <b style="color:red;">Correlation Matrix Heatmap</b>

# In[65]:


corr_matrix = numeric_df.corr()
corr_matrix


# In[66]:


plt.figure(figsize = (20, 6))
sns.heatmap(data = corr_matrix, annot = True, cmap = "coolwarm", fmt = ".2f")
plt.title("Correlation Matrix Heatmap",fontsize=16, fontweight='bold')
plt.show()


# <h2 style="color:orange;">Histograms for the columns with a significant number of Zeros</h2>

# <b style="color:red; background-color:yellow">Number of Companies Worked</b>

# In[67]:


plt.figure(figsize=(9, 5))

sns.histplot(df["NumCompaniesWorked"], bins=10, color="forestgreen", edgecolor="white", kde=True)

plt.title("Distribution of Number of Companies Worked At", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Number of Companies Worked", fontsize=12)
plt.ylabel("Employee Count", fontsize=12)

plt.tight_layout()
plt.show()


# <b style="color:red; background-color:yellow">Last Year Training Times</b>

# In[68]:


plt.figure(figsize=(9, 5))

sns.histplot(df["TrainingTimesLastYear"], bins=7, color="#7bc043", edgecolor="white", kde=True)

plt.title("Distribution of Training Times Last Year", fontsize=14, fontweight='bold', pad=15)

plt.xlabel("Number of Training Sessions", fontsize=11, labelpad=10)
plt.ylabel("Count", fontsize=11, labelpad=10)

plt.xticks(range(7))

plt.tight_layout()
plt.show()


# <b style="color:red; background-color:yellow">Number of Years at Company</b>

# In[69]:


plt.figure(figsize = (9, 5))
plt.title("Number of Years at Company", fontsize=14, fontweight='bold', pad=15)
sns.histplot(df["YearsAtCompany"], bins=10, color="#005b5c", edgecolor="white", kde=True)
plt.xlabel("No of Year At Company", fontsize=11, labelpad=10)
plt.show()


# <b style="color:red; background-color:yellow">Years in Current Role</b>

# In[70]:


plt.figure(figsize = (9, 5))
sns.histplot(df["YearsInCurrentRole"], bins=10, color="c", edgecolor="white", kde=True)
plt.title(" Distribution of Years in Current Role", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("No of Year in Current Role", fontsize=11, labelpad=10)
plt.show()


# <b style="color:red; background-color:yellow">Number of Years since Last Promotion</b>

# In[71]:


plt.figure(figsize = (9, 5))
sns.histplot(df["YearsSinceLastPromotion"], bins = 10, edgecolor = "w", color = "#4682b4", kde=True)
plt.title("Number of Years since Last Promotion", fontsize=14, fontweight='bold', pad=15)
plt.xlabel(" No Of Year Lost Promotion", fontsize=11, labelpad=10)
plt.show()


# <b style="color:red; background-color:yellow">Years with Current Manager</b>

# In[72]:


plt.figure(figsize = (9, 5))
sns.histplot(df["YearsWithCurrManager"], bins = 10, edgecolor = "w", color = "forestgreen", kde=True)
plt.title("Years with Current Manager", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("No of year With Current Manager", fontsize=11, labelpad=10)
plt.show()


# ____________________
# ____________________
# ____________________
# ____________________

# <h2 style="color:orange;">EDA</h2>

# <b style="color:red;">Display the first 5 rows of the dataset.</b>

# In[73]:


df.head()


# <b style="color:red;">Check the number of rows and columns.</b>

# In[74]:


df.shape


# <b style="color:red;">Get summary statistics for numerical columns.</b>

# In[75]:


df.describe().round(2)


# <b style="color:red;">Identify any missing values in the dataset.</b>

# In[76]:


df.isnull().sum()


# <b style="color:red;">Check for duplicate records.</b>

# In[77]:


df[df.duplicated()]


# <h3 style="color:orange;">Data Type Analysis</h3>

# <b style="color:red;">Display the data types of each column.</b>

# In[78]:


df.dtypes


# <b style="color:red;">Identify categorical and numerical columns separately.</b>

# In[79]:


df.select_dtypes(include=["int", "float"]).columns


# In[80]:


df.select_dtypes(exclude=["int", "float"]).columns


# <h3 style="color:forestgreen;">Univariate Analysis</h3>

# <b style="color:red;">Plot the distribution of 
#     <b style="color:forstgreen; background-color:yellow">Age</b>, 
#     <b style="color:forestgreen; background-color:yellow">Monthly Income</b>, and 
#     <b style="color:forestgreen; background-color:yellow">Total Working Years</b> using histograms.</b>

# In[81]:


plt.figure(figsize = (8, 4))
sns.histplot(df["Age"], bins = 10, edgecolor = "w", color = "#38B1E2", kde=True)
plt.title("Age Distribution")
plt.show()


# In[82]:


plt.figure(figsize = (8, 4))
sns.histplot(df["MonthlyIncome"], bins = 10, edgecolor = "w", color = "#133458", kde=True)
plt.title("Monthly Income Distribution")
plt.show()


# In[83]:


plt.figure(figsize = (8, 4))
sns.histplot(df["TotalWorkingYears"], bins = 10, edgecolor = "w", color = "#838921", kde=True)
plt.title("Total Working Years Distribution")
plt.show()


# <b style="color:red;">Analyze the frequency of categorical columns: 
#     <b style="color:forestgreen; background-color:yellow">Job Role</b>, 
#     <b style="color:forestgreen; background-color:yellow">Department</b>, and 
#     <b style="color:forestgreen; background-color:yellow">Education Field</b> using bar charts.</b>

# In[84]:


plt.figure(figsize = (8, 4))
sns.countplot(x = df["JobRole"], color = "#457B9D")
plt.title("Job Role Distribution")
plt.xticks(rotation = 40)
plt.show()


# In[85]:


plt.figure(figsize = (8, 4))
sns.countplot(x = df["Department"], color = "c")
plt.title("Department Distribution")
plt.show()


# In[86]:


plt.figure(figsize = (8, 4))
sns.countplot(x = df["EducationField"], color = "brown")
plt.title("Education Field Distribution")
plt.xticks(rotation = 40)
plt.show()


# 
# 
# <h3 style="color:orange;">Attrition Analysis</h3>

# <b style="color:red;">Find the percentage of employees who have left 
#     <b style="color:forestgreen; background-color:yellow">(Attrition = "Yes")</b> vs. those who stayed
#     <b style="color:forestgreen; background-color:yellow">(Attrition = "No")</b>.</b>

# In[87]:


left_count = df["Attrition"].value_counts()["Yes"]
stayed_count = df["Attrition"].value_counts()["No"]
total_emps = df.shape[0]


# In[88]:


left_per = round((left_count / total_emps) * 100, 2)
left_per


# In[89]:


stayed_per = round((stayed_count / total_emps) * 100, 2)
stayed_per


# <b style="color:red;">Create a count plot to visualize the Attrition distribution.</b>

# In[90]:


plt.figure(figsize = (8, 4))
sns.countplot(x = df["Attrition"], color = "g")
plt.title("Attrition Distribution")
plt.show()


# <b style="color:red;">Compare Average 
#     <b style="color:forestgreen; background-color:yellow">Monthly Income</b> of employees who left vs. those who stayed.</b>

# In[91]:


left_df = df[df['Attrition'] == 'Yes']
stayed_df = df[df['Attrition'] == 'No']


# In[92]:


avg_income_left = left_df['MonthlyIncome'].mean().round(2)
avg_income_stayed = stayed_df['MonthlyIncome'].mean().round(2)


# In[93]:


avg_income_left


# In[94]:


avg_income_stayed


# In[95]:


income_diff = avg_income_stayed - avg_income_left
income_diff.round(2)


# <h2 style="color:orange;">Bivariate Analysis & Feature Relationships</h2>

# <h3 style="color:forestgreen;">Correlation Analysis</h3>

# <b style="color:red;">Compute the correlation matrix for all numerical features.</b>

# In[96]:


corr_matrix = df.select_dtypes(include=["int", "float"]).corr()
corr_matrix


# <b style="color:red;">Visualize the heatmap of correlations.</b>

# In[97]:


plt.figure(figsize = (20, 6))
sns.heatmap(data = corr_matrix, annot = True, cmap = "Greens", fmt = ".2f")
plt.title("Correlation Matrix- Relationships B/W Features")
plt.show()


# <div style="border:2px solid black;  
#             background-color:yellow; 
#             color:forestgreen;
#             font-weight:bold;
#             white-space:pre; 
#             max-width:400px; 
#             line-height:0.5">
#     <ul>
#         <li>JobLevel and MonthlyIncome: <b style="color:blue;">0.95</b></li>
#         <li>TotalWorkingYears and MonthlyIncome: <b style="color:blue;">0.78</b></li>
#         <li>YearsAtCompany and YearsInCurrentRole: <b style="color:blue;">0.76</b></li>
#         <li>YearsWithCurrManager and YearsInCurrentRole: <b style="color:blue;">0.77</b></li>
#         <li>TotalWorkingYears and YearsAtCompany: <b style="color:blue;">0.63</b></li>
#     </ul>
# </div>

# <p style="color:blue;">The strong positive correlations suggest relationships tied to career progression, such as higher 
#     <b style="color:green; background-color:yellow">job levels</b>, 
#     <b style="color:green; background-color:yellow">longer tenure</b>, and 
#     <b style="color:green; background-color:yellow">increased income</b>. Weak or negligible correlations indicate that certain features may not significantly impact other variables in this dataset. From this insights, we could do further analysis, such as identifying factors influencing employee satisfaction or retention.</p>

# <h3 style="color:orange;"> Attrition vs. Numeric Features</h3>

# <b style="color:red;">Compare 
#     <b style="color:green; background-color:yellow">Monthly Income</b>, 
#     <b style="color:green; background-color:yellow">Total Working Years</b>, and 
#     <b style="color:green; background-color:yellow">Age</b> distributions for employees who left vs. stayed.</b>

# In[98]:


left_df


# In[99]:


stayed_df


# In[100]:


plt.figure(figsize = (20, 6))

# Monthly Income
plt.subplot(1, 3, 1)
plt.hist(left_df["MonthlyIncome"], bins = 20, alpha = 0.5, 
         color = "#004f15", label = "Left")
plt.hist(stayed_df["MonthlyIncome"], bins=20, alpha = 0.5, 
         color = "#e7f705", label = "Stayed")
plt.title("Monthly Income Distribution")
plt.legend()

# Total Working Years

plt.subplot(1, 3, 2)
plt.hist(left_df["TotalWorkingYears"], bins = 20, alpha = 0.5, 
         color = "#004f15", label = "Left")
plt.hist(stayed_df["TotalWorkingYears"], bins = 20, alpha = 0.5, 
         color = "#e7f705", label = "Stayed")
plt.title("Total Working Years Distribution")
plt.legend()

#Age
plt.subplot(1, 3, 3)
sns.histplot(data=df, x="Age", hue="Attrition", bins=20, kde=True, 
             palette={"Yes": "#004f15", "No": "#e7f705"}, alpha=0.5, element="step")
# 3. Clear Titles aur Axis Labels
plt.title("Age Distribution by Employee Attrition", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Age of Employees", fontsize=12)
plt.ylabel("Number of Employees", fontsize=12)
plt.tight_layout()
plt.show()


# <div style="border:2px solid black; 
#             background-color:yellow;
#             padding:5px;
#             color:purple;
#             max-width:600px; 
#             line-height:1.5;
#             word-wrap: break-word;">
#     <ul>
#         <li><b style="color:green;">Monthly Income</b>: Employees who left seem to have lower income distributions compared to those who stayed.</li>
#         <li><b style="color:green;">Total Working Years</b>: Employees with fewer working years seem more likely to leave.</li>
#         <li><b style="color:green;">Age</b>: Younger employees have higher attrition, while older employees tend to stay.</li>
#     </ul>
# </div>

# <b style="color:red;">Use violin plots or box plots for better visualization.</b>

# In[101]:


plt.figure(figsize = (20, 6))

# Monthly Income
plt.subplot(1, 3, 1)
sns.violinplot(data = df, x = "Attrition", y = "MonthlyIncome", 
               hue = "Attrition", palette = ["#ecfc05", "#0df505"])
plt.title("Monthly Income Distribution")

# Total Working Years
plt.subplot(1, 3, 2)
sns.violinplot(data = df, x = "Attrition", y = "TotalWorkingYears", 
               hue = "Attrition", palette = ["#ecfc05", "#0df505"])
plt.title("Total Working Years Distribution")

# Age
plt.subplot(1, 3, 3)
sns.violinplot(data = df, x = "Attrition", y = "Age", 
               hue = "Attrition", palette = ["#ecfc05", "#0df505"])
plt.title("Age Distribution")

plt.show()


# <div style="border:2px solid black; 
#             padding:5px;
#             background-color:yellow;
#             color:purple;
#             max-width:600px; 
#             line-height:1.5;
#             word-wrap: break-word;">
#     <ul>
#         <li><b style="color:green;">Monthly Income</b>: Clear gap in distribution—lower-income employees are more likely to leave.</li>
#         <li><b style="color:green;">Total Working Years</b>: Employees with shorter careers show higher attrition, reinforcing the histogram trend.</li>
#         <li><b style="color:green;">Age</b>: The younger demographic faces more attrition, possibly due to career shifts or better opportunities.</li>
#     </ul>
# </div>

# <b style="border:4px solid green; 
#           padding:5px;
#           color:red;">Low salary, fewer years of experience, and younger age = Higher Attrition Risk</b>

# <h3 style="color:orange;">Attrition vs. Categorical Features</h3>

# <b style="color:purple;">Create count plots for categorical variables like 
#     <b style="color:green; background-color:yellow">Job Role</b>, 
#     <b style="color:green; background-color:yellow">Department</b>, 
#     <b style="color:green; background-color:yellow">Education Field</b> segmented by 
#     <b style="color:green; background-color:yellow">Attrition</b>.</b>

# In[102]:


plt.figure(figsize = (20, 6))

# Job Role
plt.subplot(1, 3, 1)
sns.countplot(data = df, x = "JobRole", 
               hue = "Attrition", palette = ["#ecfc05", "#0df505"])
plt.title("Job Role Distribution")
plt.xticks(rotation = 90)

# Department
plt.subplot(1, 3, 2)
sns.countplot(data = df, x = "Department", 
               hue = "Attrition", palette = ["#ecfc05", "#0df505"])
plt.title("Department Distribution")
plt.xticks(rotation = 90)

# Education Field
plt.subplot(1, 3, 3)
sns.countplot(data = df, x = "EducationField", 
               hue = "Attrition", palette = ["#ecfc05", "#0df505"])
plt.title("Education Field Distribution")
plt.xticks(rotation = 90)

plt.show()


# 
# <b style="color:green; background-color:yellow">Job Role</b>
# 
# <div style="border:2px solid black; 
#             padding:5px;
#             background-color:yellow;
#             color:purple;
#             max-width:600px; 
#             line-height:1.5;
#             word-wrap: break-word;">
#     <ul>
#         <li>
#             <b style="color:green;">Sales Executive and Research Scientist</b> roles have the highest counts of employees, but attrition is relatively higher in these roles compared to others.</li>
#         <li>
#             <b style="color:green;">Laboratory Technician</b> also shows a significant number of employees leaving, indicating potential dissatisfaction or challenges in this role.</li>
#         <li>Roles like 
#             <b style="color:green;">Manager, Research Director, and Healthcare Representative</b> have lower attrition rates, suggesting better retention in these positions.</li>
#     </ul>
# </div>

# <b style="color:green; background-color:yellow">Department</b>
# 
# <div style="border:2px solid black; 
#             padding:5px;
#             background-color:yellow;
#             color:purple;
#             max-width:600px; 
#             line-height:1.5;
#             word-wrap: break-word;">
#     <ul>
#         <li>The 
#             <b style="color:green;">Research & Development</b> department has the largest workforce, but it also experiences substantial attrition, although the majority of employees stay.</li>
#         <li>The 
#             <b style="color:green;">Sales </b> department shows a relatively higher proportion of attrition compared to its total workforce.</li>
#         <li>The 
#             <b style="color:green;">Human Resources</b> department has the smallest workforce and exhibits minimal attrition.</li>
#     </ul>
# </div>

# <b style="color:green; background-color:yellow">Education Field</b>
# 
# <div style="border:2px solid black; 
#             padding:5px;
#             background-color:yellow;
#             backgroun-color:yellow;
#             color:purple;
#             max-width:600px; 
#             line-height:1.5;
#             word-wrap: break-word;">
#     <ul>
#         <li>Employees with a background in 
#             <b style="color:green;">Life Sciences and Medical</b> fields dominate the workforce. However, attrition is more noticeable in these fields, particularly in 
#             <b style="color:green;">Life Sciences</b>.</li>
#         <li>Fields like 
#             <b style="color:green;">Marketing, Technical Degree, and Human Resources</b> exhibit lower attrition rates, possibly indicating better alignment with job roles or satisfaction levels.</li>
#     </ul>
# </div>

# <b style="border:4px solid green; 
#           padding:5px;
#           color:forestgreen;">Retention efforts should focus on Sales, R&D, and employees from Medical/Life Sciences backgrounds.</b>

# In[ ]:




