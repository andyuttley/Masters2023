import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import requests
from bs4 import BeautifulSoup


# Set the title and icon of the app
st.set_page_config(page_title="Masters 2023 Predictor", page_icon=":golf:", initial_sidebar_state="expanded", page_bg_color="white")



# scrape stats
# Fetch the website content
url = 'https://www.espn.com/golf/rankings'
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the rankings
table = soup.find_all('table')[0]

# Extract the table rows and convert them into a list of dictionaries
rows = table.find_all('tr')[1:]
data = []
for row in rows:
    cols = row.find_all('td')
    rank = cols[0].text.strip()
    name = cols[1].text.strip()

    data.append({
        'Rank': rank,
        'Name': name,

    })

# Convert the list of dictionaries into a pandas dataframe
df_rankings = pd.DataFrame(data)




linkedinlink = '[Andy Uttley - LinkedIn](https://www.linkedin.com/in/andrewuttley/)'
mediumlink = '[Andy Uttley - Medium Blog](https://andy-uttley.medium.com/)'

#Create header
st.write("""# MASTERS 2023 Predictor""")
st.write("""## How it works""")
st.write("Model your predicted winner by using the left side of the screen to apply weightings to different key metrics."
         "The current selections are those deemed most appropriate to the Masters based on recent outcomes.")





#Bring in the data
data = pd.read_excel('PGA_Database_2023_4.xlsx')
pca_data = pd.read_excel('PGA_Database_2023_pca.xlsx')

#Create and name sidebar
st.sidebar.header('Choose your weightings')

st.sidebar.write("""#### Choose your SG bias""")
def user_input_features():
    sgott = st.sidebar.slider('SG Off the Tee', 0, 100, 40, 5)
    sgt2g = st.sidebar.slider('SG Tee to Green', 0, 100, 60, 5)
    sga2g = st.sidebar.slider('SG Approach to Green', 0, 100, 90, 5)
    sgatg = st.sidebar.slider('SG Around the Green', 0, 100, 70, 5)
    sgputt = st.sidebar.slider('SG Putting', 0, 100, 80, 5)
    sgmasters = st.sidebar.slider('SG Masters History', 0, 100, 50, 5)
    sgtotal = st.sidebar.slider('SG Total', 0, 100, 30, 5)
    sgpar5 = st.sidebar.slider('SG Par 5s', 0, 100, 85, 5)
    sgpar4 = st.sidebar.slider('SG Par 4s', 0, 100, 30, 5)
    sgpar3 = st.sidebar.slider('SG Par 3s', 0, 100, 30, 5)



    user_data = {'SG OTT': sgott,
                     'SG T2G': sgt2g,
                     'SG A2G': sga2g,
                     'SG ATG': sgatg,
                     'SG Putt': sgputt,
                     'SG Total': sgtotal,
                     'SG Par 5': sgpar5,
                     'SG Par 4': sgpar4,
                     'SG Par 3': sgpar3,
                     'SG Masters': sgmasters}
    features = pd.DataFrame(user_data, index=[0])
    return features

df_user = user_input_features()

if st.sidebar.checkbox("Choose a recency bias"):
    def user_input_biased():
        thisyear = st.sidebar.slider('2022 weighting', 0, 100, 100, 5)
        lastyear = st.sidebar.slider('2021 weighting', 0, 100, 40, 5)
        biased_data = {'this year': thisyear/100,
                       'last year': lastyear/100}
        biased = pd.DataFrame(biased_data, index=[0])
        return biased


    df_user_biased = user_input_biased()

else:
    def user_input_biased():
        thisyear = 100
        lastyear = 40
        biased_data = {'this year': thisyear / 100,
                       'last year': lastyear / 100}
        biased = pd.DataFrame(biased_data, index=[0])
        return biased
    df_user_biased = user_input_biased()


st.write("## CURRENT CHOSEN WEIGHTINGS: ")
df_user



def results_output():
    sg_ott = (data['SG_OTT_2023']*df_user_biased['this year'][0] + data['SG_OTT_2022']*df_user_biased['last year'][0]) * df_user['SG OTT'][0] / 100
    sg_t2g = (data['SG_TeeToGreen_2023']*df_user_biased['this year'][0] + data['SG_TeeToGreen_2022']*df_user_biased['last year'][0]) * df_user['SG T2G'][0] / 100
    sg_a2g = (data['SG_A2G_2023']*df_user_biased['this year'][0]  + data['SG_A2G_2022']*df_user_biased['last year'][0]) * df_user['SG A2G'][0] / 100
    sg_atg = (data['SG_ATG_2023']*df_user_biased['this year'][0]  + data['SG_ATG_2022']*df_user_biased['last year'][0]) * df_user['SG ATG'][0] / 100
    sg_total = (data['SG_Total_2023']*df_user_biased['this year'][0]  + data['SG_Total_2022']*df_user_biased['last year'][0]) * df_user['SG Total'][0]/100
    sg_putt = (data['SG_Putting2023']*df_user_biased['this year'][0]  + data['SG_Putting2022']*df_user_biased['last year'][0]) * df_user['SG Putt'][0]/100
    #SG Par requires additional logic (par 5 2022 needs adding to file)
    sgpar5 = (5 - data['Par5ScoringAvg_2023'] * df_user_biased['this year'][0] + 5 - data['Par5ScoringAvg_2022'] * df_user_biased['last year'][0]) * df_user['SG Par 5'][0] / 100
    sgpar4 = (4 - data['Par4ScoringAvg_2023'] * df_user_biased['this year'][0] + 4 - data['Par4ScoringAvg_2022'] * df_user_biased['last year'][0]) * df_user['SG Par 4'][0] / 100
    sgpar3 = (3 - data['Par3ScoringAvg_2023'] * df_user_biased['this year'][0] + 3 - data['Par3ScoringAvg_2022'] * df_user_biased['last year'][0]) * df_user['SG Par 3'][0] / 100
    #SG Masters diff calc
    sgmasters = (data['MastersSG']*((df_user_biased['last year'][0] + df_user_biased['this year'][0])/2) * df_user['SG Masters'][0]/100)

    results = {'Name': data['PLAYER']
               , 'Total SG per round': (sg_ott + sg_t2g + sg_a2g + sg_atg + sg_putt + sgpar5 + sgpar4 + sgpar3 + sgmasters + (sg_total/9))
               , 'SG OTT Weighted': sg_ott
               , 'SG T2G Weighted': sg_t2g
               , 'SG A2G Weighted': sg_a2g
               , 'SG ATG Weighted': sg_atg
               , 'SG Putt Weighted': sg_putt
               , 'SG Par 5 Weighted': sgpar5
                , 'SG Par 4 Weighted': sgpar4
                , 'SG Par 3 Weighted': sgpar3
                , 'SG Masters': sgmasters
                 , 'SG Total': sg_total
               }
    resultpd = pd.DataFrame(results)
    resultpd.sort_values(by=['Total SG per round'], ascending=False, inplace=True)
    return resultpd

df_results  = results_output()

#Output rankings based on users selections
st.write(
    """
    ## CURRENT PREDICTION OUTPUT
    """
)

# use softmax to create the % probability
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0))*100

df_results['prediction'] = softmax(df_results['Total SG per round'])
df_results2 = df_results[['Name', 'prediction', 'Total SG per round']]
df_results2['Est. Odds'] = ((100/df_results2['prediction']).astype(int)).astype(str)+'/1'
df_results2.reset_index(inplace=True)
df_results2.drop('index', axis=1, inplace=True)

winner = df_results2['Name'][0]
predperc = df_results2['prediction'][0]
st.markdown(f"The predicted winner is **{winner:}** who has a **{predperc:.2f}**% chance of winning")


#image of winner

try:
    winnerimage = Image.open(winner+'.jpg')
    st.image(winnerimage)
except:
    pass


st.write("## Full table of results")
st.write("Results are calculated using the weightings you have applied to historic scraped player data, and uses softmax to create a prediction. Softmax exponentiates each number and then divides each exponentiated value by the sum of all the exponentiated values. The resulting values are always between 0 and 1 and sum up to 1 (100% chance one of them will win), which makes them useful for representing probabilities.")


df_results2 = pd.merge(df_results2, df_rankings[['Name', 'Rank']], on='Name', how='left')
df_results2['Rank'] = pd.to_numeric(df_results2['Rank'], errors='coerce')
df_results2['Rank in Masters'] = df_results2['Rank'].rank(method='min', ascending=True)
df_results2 = df_results2.rename(columns={'Rank': 'Current World Rank'})
df_results2['Current World Rank'] = pd.to_numeric(df_results2['Current World Rank'], errors='coerce')
df_results2['Current World Rank'] = df_results2['Current World Rank'].fillna(72)
df_results2['Rank in Masters'] = pd.to_numeric(df_results2['Rank in Masters'], errors='coerce')
df_results2['Rank in Masters'] = df_results2['Rank in Masters'].fillna(72)
df_results2['Rank Difference'] = df_results2['Rank in Masters'] - (df_results2.index + 1)
# Get column list
cols = list(df_results2.columns)

# Move Current World Rank to end of list
cols.append(cols.pop(cols.index('Current World Rank')))

# Reorder columns in dataframe
df_results2 = df_results2[cols]
st.dataframe(df_results2.drop('Total SG per round', axis=1))



# create bar chart
st.write("## YOUR RANKED RESULTS OF TOP 40")

chart = alt.Chart(df_results2[:40]).mark_bar().encode(
    x=alt.X("prediction",sort=None),
    y=alt.Y('Name', sort=None),
    opacity=alt.value(1),
color=alt.condition(
    alt.datum.Name == df_results2['Name'][0],  # If it's the top ranked prediction
        alt.value('#f63366'),     #  sets the bar to the streamlit pink.
        alt.value('grey')  ) # else this colour
).properties(
    width=380
)


text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=3  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text=alt.Text('prediction', format=',.2r')
)

st.altair_chart(chart+text)


st.write("### PCA Chart")
st.write("Principal Component Analysis (PCA) is a statistical technique used to reduce the dimensionality of a dataset by identifying the most important features that explain the variation in the data. It can be used to compare golfers by identifying the underlying patterns in their game data and finding similarities between them based on these patterns.")
#####PCA CHART

# Merge dataframes on 'PLAYER' column
merged_data = pd.merge(pca_data, df_results2, left_on='PLAYER', right_on='Name')

# Extract final word from PLAYER column
merged_data['label'] = merged_data['PLAYER'].apply(lambda x: x.split()[-1])

# Define custom colorscale
colorscale = [[0.0, '#9E1B32'], [0.5, '#F1BF00'], [1.0, '#009F3D']]

# Define scatter plot using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_data['PC1'], y=merged_data['PC2'], mode='markers+text', text=merged_data['label'],
                         marker=dict(color=merged_data['prediction'], colorscale=colorscale,
                                     size=merged_data['prediction'] * 2, sizemode='diameter', sizemin=5)
                         ,
                         textposition='bottom center', textfont=dict(size=10, color='white')))
fig.update_layout(width=800, height=600, title="PCA Scatter Plot")

# Display the plot using Streamlit
st.plotly_chart(fig)



st.write("## For more information visit:")
st.write(mediumlink, " | ", linkedinlink)


#image
image = Image.open('Tiger.jpg')
st.image(image)


showdata = st.checkbox('Show underlying data being used', value=False)
if showdata:
        st.write("Data taken from multiple PGA stat URLs. Imputation is used in the case of missing data.")
        data


#TO DO
## players in masters
## par 5 last year
## masters history
## pca scatter
## simulation? 100k masters
## better logic for prediction?



