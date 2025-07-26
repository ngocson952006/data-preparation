import pandas as pd
from IPython.core.pylabtools import figsize
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

DATA_DIR = "data-files"
pd.set_option('display.max_columns', None)


def profile_titanic(filename):
    print("Start profiling data...")
    df = pd.read_csv(f"{DATA_DIR}/{filename}")
    profile = ProfileReport(df, title="Titanic Data Profile")
    profile.to_file(f"{DATA_DIR}/{filename}-profile.html")
    print("Finish profiling data")


def initialize_df(filename):
    df = pd.read_csv(f"{DATA_DIR}/{filename}")
    df['SurvivedLabel'] = df['Survived'].map({0: 'not alive', 1: 'alive'})
    return df


def analyze_survival_by_sex(df):
    """
    Analyzes and visualizes the count of survival outcomes on the Titanic based on gender.
    Groups the data by sex and survival status, plots a stacked bar chart showing the
    distribution of survivors and non-survivors for each gender.

    :param df: DataFrame containing Titanic passenger data with at least the following
        columns:
            - 'Sex': The gender of the passenger (e.g., 'male', 'female').
            - 'Survived': Survival status, where 1 indicates survival, and 0 indicates
              non-survival.
    :return: None
    """
    survival_count = df.groupby(['Sex', 'Survived']).size().unstack()
    survival_count.plot(kind='bar', stacked=False, figsize=(6, 4))
    plt.title('Survival Counts by Sex on the Titanic')
    plt.xlabel('Sex')
    plt.ylabel('Number of Passengers')
    plt.xticks(rotation=0)
    plt.legend(['Did Not Survive', 'Survived'], title='Outcome')
    plt.show()


def analyze_survival_by_age(df):
    """
    Analyze survival data based on passenger age.

    This function analyzes the survival data of passengers by considering their
    age. It calculates various statistical measures (mean, median, 1st
    quartile, 3rd quartile) for the Age column and visualizes the
    distribution of ages for all passengers and survivors in a histogram.

    The function also drops rows with missing age values before proceeding
    with calculations and plots.

    :param df: Pandas DataFrame containing passenger data. It must have 'Age' and
        'SurvivedLabel' columns.
    :type df: pandas.DataFrame
    :return: None
    :rtype: None
    """
    print("Start analyzing survival by age...")
    print(df['Age'].isnull().sum())
    with_age = df.dropna(subset=['Age'])  # create new dataset that having age value filled
    # Get means of age variable
    age_mean = with_age['Age'].mean()
    age_median = with_age['Age'].median()
    age_q1 = with_age['Age'].quantile(0.25)
    age_q3 = with_age['Age'].quantile(0.75)
    print(f"Age mean: {age_mean}")
    print(f"Age median: {age_median}")
    print(f"Age Q1: {age_q1}")
    print(f"Age Q3: {age_q3}")

    # Start creating the pilot
    plt.figure(figsize=(14, 6))
    age_survivants = with_age[with_age['SurvivedLabel'] == 'alive']['Age']  # filter the ages of alive passengers
    plt.hist(with_age['Age'], bins=20, alpha=0.7, label='All Passengers')
    plt.hist(age_survivants, bins=20, alpha=0.7, label='Survivants')
    plt.axvline(age_mean, color='y', linestyle='--', label='Mean (mean)')
    plt.axvline(age_q1, color='r', linestyle='--', label='Q1 (25% quantile)')
    plt.axvline(age_median, color='b', linestyle='--', label='Median (50%)')
    plt.axvline(age_q3, color='g', linestyle='--', label='Q3 (75% quantile)')
    plt.title('Number of passengers by Age', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of Passengers', fontsize=14)
    plt.legend(loc='upper right')
    plt.show()


def analyze_survival_by_age_groups_and_genders(df):
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
    # start grouping the dataframe
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    survival_by_gender_age_group = df.groupby(['Sex', 'AgeGroup', 'Survived']).size().unstack(fill_value=0)
    survival_by_gender_age_group['SurvivalRate'] = survival_by_gender_age_group[1] / (
            survival_by_gender_age_group[1] + survival_by_gender_age_group[0])
    female_survival_by_age_group = survival_by_gender_age_group.loc['female']['SurvivalRate']
    male_survival_by_age_group = survival_by_gender_age_group.loc['male']['SurvivalRate']

    # start plotting the graph
    bar_width = 0.35
    x = np.arange(len(age_labels))
    print(x)
    plt.figure(figsize=(14, 6))
    plt.title('Survival Rate by Age Group and Gender')
    plt.xlabel('Age Group')
    plt.ylabel('Survival Rate')
    plt.bar(x - bar_width / 2, female_survival_by_age_group, width=bar_width, label='Female Survival Rate')
    plt.bar(x + bar_width / 2, male_survival_by_age_group, width=bar_width, label='Male Survival Rate')
    plt.xticks(x, age_labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def analyze_survival_by_group_rate(data_frame, col_name):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=col_name,
                  hue='Survived',
                  data=data_frame[data_frame['Survived'].notnull()],
                  ax=axs[0])
    axs[0].set_title("Survival by Group")
    axs[0].set_xlabel("Group")
    axs[0].set_ylabel("Count")

    survival_rates = data_frame.groupby(col_name)['SurvivedProba'].mean()
    print(survival_rates)
    sns.barplot(x=survival_rates.index, y=survival_rates.values, ax=axs[1])
    axs[1].set_title('Survival Rate by Group')
    axs[1].set_xlabel('Title Group')
    axs[1].set_ylabel('Survival Rate')
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()



def analyze_ticket_counts(df):
    ticket_counts = df.groupby('Ticket')['PassengerId'].count().reset_index()
    # Find tickets that have more than passenger
    same_tickets = ticket_counts[ticket_counts['PassengerId'] > 1]
    analyze_df = df.merge(same_tickets[['Ticket']], on='Ticket', how='inner')
    analyze_df[["PassengerId", "Ticket", "Fare"]].sort_values(by=["Ticket"])
    print(analyze_df)


def analyze_fare_and_ticket(df):
    ticket_fare = df.groupby('Ticket')['Fare'].unique().reset_index()
    # Find the ticket that has the same fare at all
    ticket_fare['SameFare'] = ticket_fare['Fare'].apply(lambda x: len(set(x)) == 1)
    print(ticket_fare)
    analyze_df = df.merge(ticket_fare[['Ticket', 'SameFare']], on='Ticket', how='inner')
    # count values for each SameFare and report
    print(analyze_df['SameFare'].value_counts(normalize=True))


def name_to_word_pattern(name):
    name_without_parentheses = re.sub(r'\([^)]*\)', '', name).strip()
    name_to_words = re.findall(r'\b\w+\b|\.|,', name_without_parentheses)
    patterns = []
    for word in name_to_words:
        if word in [',', '.']:
            patterns.append(word)
        elif word.lower() in ['mr', 'mrs', 'miss', 'master', 'don', 'rev', 'dr', 'ms', 'sir', 'lady', 'major', 'capt',
                              'col']:
            patterns.append('TITLE')
        elif word.lower() in ['van', 'de', 'der', 'du', 'di', 'la', 'le']:
            patterns.append('PREFIX')
        else:
            if not patterns or patterns[-1] != 'NAME':
                patterns.append('NAME')
    return ' '.join(patterns)


def parse_name(name):
    name_without_parentheses = re.sub(r'\([^)]*\)', '', name).strip()
    name_to_words = re.findall(r'\b\w+\b|\.|,', name_without_parentheses)
    prefix = None
    title = None
    names = []
    for word in name_to_words:
        if word.lower() in ['mr', 'mrs', 'miss', 'master', 'don', 'rev', 'dr', 'ms', 'sir', 'lady', 'major', 'capt',
                            'col']:
            title = word.lower()
        elif word.lower() in ['van', 'de', 'der', 'du', 'di', 'la', 'le']:
            prefix = word.lower()
        elif word not in ['.', ',']:
            names.append(word)
    return {
        'prefix': prefix,
        'title': title,
        'names': names
    }

def categorize_title(title):
    """ This function categorizes a given title into predefined groups based on its
        social or professional context. Titles are grouped as 'Common', 'Rich', or
        'Professional', while unrecognized titles are assigned a NaN value.
    Parameters:
        title (str): The input title to categorize (e.g., 'Mr', 'Dr', 'Lady').
    Returns:
        str or np.nan: The category of the title:
            - 'Common' for widely used social titles (e.g., 'Mr', 'Miss').
            - 'Rich' for titles typically associated with wealth or nobility
            (e.g., 'Don', 'Lady').
            - 'Professional' for titles related to professions or ranks
            (e.g., 'Dr', 'Major').
            - np.nan if the title does not match any predefined categories.
    """
    if title is None:
        return np.nan

    if title.lower() in ['mr', 'mrs', 'ms', 'miss', 'mme', 'mlle']:
        return 'Common'
    elif title.lower() in ['master', 'don', 'lady', 'sir', 'jonkheer', 'dona']:
        return 'Rich'
    elif title.lower() in ['rev', 'dr', 'major', 'col', 'capt']:
        return 'Professional'
    else:
        return np.nan

if __name__ == "__main__":
    # profile_titanic("titanic-passengers.csv")
    df = initialize_df("titanic-passengers.csv")
    # print(df['Sex'].value_counts())
    # #correlation_pilot_by_columns(df, "Age", "Survived")
    # analyze_survival_by_sex(df)
    # analyze_survival_by_age(df)
    # analyze_survival_by_age_groups_and_genders(df)
    # analyze_fare_and_ticket(df)
    df['NamePattern'] = df['Name'].apply(name_to_word_pattern)
    df['NameComponents'] = df['Name'].apply(parse_name)
    # apply name components to particular columns
    df['Title'] = df['NameComponents'].apply(lambda x: x['title'])
    df['Prefix'] = df['NameComponents'].apply(lambda x: x['prefix'])
    df['Names'] = df['NameComponents'].apply(lambda x: x['names'])
    df['TitleGroup'] = df['Title'].apply(categorize_title)
    df = df.drop(columns=['NameComponents'])
    # filter df where Prefix is not None
    df['SurvivedProba'] = df['Survived']
    print(df)
    analyze_survival_by_group_rate(df, "TitleGroup")
