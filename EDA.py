import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import seaborn as sns 
from wordcloud import WordCloud
import re

#import dataset 
loan = pd.read_csv('loan.csv', low_memory=False)

########################
#distribution of loan 
########################
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16,8))

tempStatus = loan['loan_status'].value_counts()
tempStatus.sort_values(ascending=False, inplace=True)
sns.barplot(x=tempStatus.values, y=tempStatus.index, palette='GnBu_d', orient='h', ax=ax1)
ax1.set_xscale('log')
for p, ratio in zip(ax1.patches, [str(round((100*x/len(loan)), 1))+'%' for x in tempStatus.values]):
    ax1.text(p.get_width() + p.get_x(),
             p.get_y() + p.get_height()/1.3,
             ratio,
             ha='center')
ax1.set_xlabel('Status count')
ax1.set_ylabel('Loan status')
ax1.set_title('Loan status distribution')

loan['year'] = [re.split('-', x)[-1] for x in loan['issue_d']]

#define bad loan conditions
bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", 
            "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]
loan['loan_good_bad'] = loan['loan_status'].apply(lambda x: 'bad_loan' if x in bad_loan else 'good_loan')

sns.barplot(x="year", y="loan_amnt", hue="loan_good_bad", data=loan,
            estimator=lambda x: len(x) / len(loan) * 100, ax=ax2)
ax2.set_title('Loan conditions over years')
ax2.set_ylabel('Percentage')
ax2.set_xlabel('Year')
plt.show()

################
#purpose of loan
################
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,8))

tempPurpose = loan['purpose'].value_counts()
tempPurpose.sort_values(ascending=False, inplace=True)
sns.barplot(x=tempPurpose.values, y=tempPurpose.index, ax=ax1)
for p, ratio in zip(ax1.patches, [str(round(100*x/len(loan),2))+'%' for x in tempPurpose.values]):
    ax1.text(p.get_width() + p.get_x(),
           p.get_height()/1.3 + p.get_y(),
           ratio,
           ha='center')
ax1.set_xscale('log')
ax1.set_xlabel('Count')
ax1.set_ylabel('Purpose')
ax1.set_title('Loan purpose distribution')

wc = WordCloud(background_color=None, mode='RGBA', relative_scaling=0.2, normalize_plurals=False).generate(str(loan['title']))
ax2.imshow(wc, interpolation='bilinear')
ax2.set_title('Word cloud of loan purpose')
plt.show()

########################
#loan amount distribution
########################
fig, ax = plt.subplots(2, 2, figsize=(16,8), squeeze=False)
plt.subplots_adjust(hspace=0.4)

tempLoanBorrower = loan['loan_amnt']
tempLoanLender = loan['funded_amnt']
tempLoanInv = loan['funded_amnt_inv']
tempIssueDate = pd.to_datetime('01-' + loan['issue_d'])
tempIssue = loan['funded_amnt'].groupby(tempIssueDate).sum()

sns.distplot(a=tempLoanBorrower, ax=ax[0,0])
ax[0,0].set_title('Loan amount applied by borrower')

sns.distplot(a=tempLoanLender, ax=ax[0,1])
ax[0,1].set_title('Loan amount funded by lender')

sns.distplot(a=tempLoanInv, ax=ax[1,0])
ax[1,0].set_title('Amount committed by investors')

ax[1,1].plot_date(x=tempIssue.index, y=tempIssue.values, linestyle='-', marker='')
ax[1,1].set_title('Total amount of loan funded over time')
ax[1,1].set_xlabel('Date')
ax[1,1].set_ylabel('Total amount funded')
#ax[1,1].set_yscale('log')

plt.show()
