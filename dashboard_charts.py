
import altair as alt
import pandas as pd

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import math

import streamlit as st

from dashboard_data import *

pio.templates.default = "simple_white"

#from streamlit_plotly_events import plotly_events

ge_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
carer_labels = ['anger','fear','joy','love','sadness','surprise']
isear_labels = ['anger','disgust','fear','guilt','joy','sadness','shame']
se_labels = ['anger','joy','others','sadness']
tec_labels = ['anger','disgust','fear','joy','sadness','surprise']

nrc_labels = ['nrc_anger', 'nrc_disgust', 'nrc_fear', 'nrc_joy', 'nrc_sadness', 'nrc_surprise']
liwc_emotion_labels = ['liwc_affect','liwc_posemo','liwc_negemo','liwc_anx','liwc_anger','liwc_sad']
liwc_other_labels = ['liwc_negate','liwc_social','liwc_cogmech','liwc_percept','liwc_bio','liwc_relativ','liwc_relig','liwc_death']



@st.cache_data
def get_comments(feature_option):
    if feature_option == '#Lines':
        return ['This measure is important to identify the most relevant text sizes in the corpus. Each line of the dataset has a text with a specific size.']
    elif feature_option == '#Duplicated Texts':
        return ['#The duplicated texts may disturb the machine learning task.']
    elif feature_option == 'Top Words':
        return ['Not done yet.']
    elif feature_option == '#Entities':
        return ['Not done yet.']
    elif feature_option == 'Entities':
        return ['The top most entities related to datasets or emotions will be plotted.','Group by emotions option, text size filter and showing texts are unavailable.']    
    elif feature_option == 'Average #Sentences/Line':
        return ["The #Sentences/Line according to text size can help to identify which emotion classes are the most expressed by sentences' concatenations."]
    elif feature_option == 'Average #Words/Line':
        return ['The #Words/Line according to text size is important to know how many elements are used to express the emotion.']
    elif feature_option == 'Average TTR/Line':
        return ['TTR = the total number of unique words / the total number of words.','A high TTR indicates a high degree of lexical variation.']
    elif feature_option == 'Average MTLD/Line':
        return ['Measure of Textual Lexical Diversity (MTLD) tries to account for and avoid the influence of text size.','This measure is the average length of the longest consecutive sequences of words that maintain a TTR of at least min_ttr (0.72 by default).']
    elif feature_option == 'Avg. Flesch-Kincaid Grade Level/Line':
        return ['The "Flesch–Kincaid Grade Level Formula" presents a score as a U.S. grade level to measure the readability level of various books and texts.']
    elif feature_option == 'Avg. Flesch Reading Ease/Line':
        return ['The Flesch Reading Ease is a readability score. Typically, a score of 100 or more means that the content is very simple and easy to read.','According to textacy library, the score may be arbitrarily negative in extreme cases.']
    elif feature_option == 'Pearson':
        return ['The Pearson correlation measures the strength of the linear relationship between two variables. A value between 0.7 and 0.9 can be considered a high correlation. A score between 0.5 and 0.7 indicates a moderate correlation.']
    elif feature_option == 'Jaccard':
        return ['The Jaccard similarity coefficient, is a metric used for gauging the similarity and diversity of sample sets.','Its value is the size of the intersection divided by the size of the union of the sample sets.']
    elif feature_option == 'Dice':
        return ['The Sørensen–Dice coefficient is a metric used to gauge the similarity of two samples.','It is considered a semimetric version of the Jaccard index.']
    else:
        return []

    

@st.cache_data
def get_distfreq_chart(df, datasetname, chart_width, chart_height):
    if datasetname == 'All Datasets':
        c = alt.Chart(df, title="Emotion Frequency Distribution", width=chart_width, height=chart_height).mark_bar().encode(
            x=alt.X('counts:Q', title='#Lines'),
            y=alt.Y('emotion:N', title='Emotion'),
            color=alt.Color('source:N', 
                            scale=alt.Scale(
                                domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])),
            order=alt.Order('source:N', sort='ascending'),
            tooltip = [
                alt.Tooltip('emotion:N',title='Emotion'),
                alt.Tooltip('source:N',title='Dataset'),
                alt.Tooltip('counts:Q',title='#Lines', format=",.0f")]
        )

    else:
        chart_emotions = sorted(list(set(df.emotion.tolist())))
        chart_colors = get_emotions_colors(chart_emotions)
        emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)

        c = alt.Chart(df, title="Emotion Frequency Distribution", width=chart_width, height=chart_height).mark_bar().encode(
            x=alt.X('counts:Q', title='#Lines'),
            y=alt.Y('emotion:N', title='Emotion'),
            #color=alt.Color('emotion:N',title='Emotion', scale=alt.Scale(scheme='category20')),
            color=alt.Color('emotion:N', scale=emotion_scale, title='Emotion'),
            tooltip = [
                alt.Tooltip('emotion:N',title='Emotion'),
                alt.Tooltip('counts:Q',title='#Lines', format=",.0f")]
        )
    return c

    
@st.cache_data
def get_texts_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    try:
        df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length < max_textsize)]
    except:
        df_chart = pd.DataFrame()

    if datasetname == 'All Datasets':       
        c = alt.Chart(df_chart, width=900, height=500).mark_bar(size=8).encode(
            x=alt.X('source:N',title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('count():Q',title='#Lines'),
            color=alt.Color('source:N',title=None, 
                            scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                            range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
                            ),
            column=alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=20), spacing = 20),
        ).properties(width=alt.Step(8))
        
    else:
        df_chart = df_chart[(df_chart.source == datasetname)]        
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
       
            c = alt.Chart(df_chart, title="#Texts Frequency By Emotions", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=50)),
                y=alt.Y('count()',title='#Lines'),
                color=alt.Color('emotion:N', scale=emotion_scale)
            ).properties(width=1200)

            c = alt.Chart(df_chart, title="#Texts Frequency By Emotions", width=800, height=600).mark_area(interpolate='monotone').encode(
                alt.X("length:Q", title='text length', 
                    axis=alt.Axis(domain=False, tickSize=0), bin=alt.Bin(maxbins=40)
                ),
                alt.Y('count():Q', stack='center', axis=None),
                alt.Color('emotion:N',
                    scale=emotion_scale
                )
            )


        else:
            c = alt.Chart(df_chart, title="#Texts Frequency", width=1000, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=50)),
                y=alt.Y('count()',title='#Lines')
            ).properties(width=1200)

    return c

@st.cache_data
def get_duplicated_texts_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df

    if datasetname == 'All Datasets':

        c = alt.Chart(df_chart, width=900, height=500, title='#Duplicated Lines (%)').transform_calculate(
            percent_dup=(100*alt.datum.n_lines_dup/alt.datum.n_lines)
        ).mark_bar(size=10).encode(
            x=alt.X('source:N',title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('percent_dup:Q',title='#Duplicated Lines (%)'),
            color=alt.Color('source:N',title=None, 
                            scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                            range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
                            ),
            column=alt.X("bin:O", title='Text size', spacing = 10),
            tooltip=[alt.Tooltip('source:N',title='Dataset'),
                     alt.Tooltip('percent_dup:Q',title='#Duplicated Lines (%)'),
                     alt.Tooltip('bin:O',title='Text size')]
        ).properties(width=alt.Step(15))

    else:
        if group_by_emotions:            
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)

            c = alt.Chart(df_chart, width=900, height=500, title='#Duplicated Lines (%)').transform_calculate(
                percent_dup=(100*alt.datum.n_lines_dup/alt.datum.n_lines)
            ).mark_bar(size=10).encode(
                x=alt.X('emotion:N',title=None, axis=alt.Axis(labels=False)),
                y=alt.Y('percent_dup:Q',title='#Duplicated Lines (%)'),
                color=alt.Color('emotion:N',title=None, scale=emotion_scale),
                column=alt.X("bin:O", title='Text size', spacing = 10),
                tooltip=[alt.Tooltip('emotion:N',title='Emotion'),
                         alt.Tooltip('percent_dup:Q',title='#Duplicated Lines (%)'),
                         alt.Tooltip('bin:O',title='Text size')]
            ).properties(width=alt.Step(15))


        else:
            c = alt.Chart(df_chart, width=900, height=500, title='#Duplicated Lines (%)').transform_calculate(
                percent_dup=(100*alt.datum.n_lines_dup/alt.datum.n_lines)
            ).mark_bar(size=20).encode(
                x=alt.X('bin:O',title='Text size'),
                y=alt.Y('percent_dup:Q',title='#Duplicated Lines (%)'),
                tooltip=[alt.Tooltip('bin:O',title='Text size'),
                         alt.Tooltip('percent_dup:Q',title='#Duplicated Lines (%)')]
            ).properties(width=alt.Step(40))
        
    return c, df_chart

@st.cache_data
def get_words_per_text_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length < max_textsize)]

    if datasetname == 'All Datasets':
        c = alt.Chart(df_chart, title="Average #Words/Line", width=900, height=500).mark_bar(size=8).encode(
            x=alt.X('source:N',title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('mean(n_words)',title='Avg #Words/Line'),
            color=alt.Color('source:N',title=None, 
                            scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                            range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
                            ),
            column=alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=20), spacing = 20),
        ).properties(width=alt.Step(8))

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
            c = alt.Chart(df_chart, title="Average #Words/Line", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=50)),
                y=alt.Y('mean(n_words)',title='Avg #Words/Line'),
                color=alt.Color('emotion:N', scale=emotion_scale)
            )
        else:
            c = alt.Chart(df_chart, title="Average #Words/Line", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=50)),
                y=alt.Y('mean(n_words)',title='Avg #Words/Line')
            )

    return c

@st.cache_data
def get_sentences_per_text_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length < max_textsize)]

    if datasetname == 'All Datasets':
        c = alt.Chart(df_chart, title="Average #Sentences/Line", width=900, height=500).mark_bar(size=8).encode(
            x=alt.X('source:N',title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('mean(n_sents)',title='Avg #Sentences/Line'),
            color=alt.Color('source:N',title=None, 
                            scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                            range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
                            ),
            column=alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=20), spacing = 20),
        ).properties(width=alt.Step(8))        

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
            c = alt.Chart(df_chart, title="Average #Sentences/Line", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=50)),
                y=alt.Y('mean(n_sents)',title='Avg #Sentences/Line'),
                color=alt.Color('emotion:N', scale=emotion_scale)
            )
        else:
            c = alt.Chart(df_chart, title="Average #Sentences/Line", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=50)),
                y=alt.Y('mean(n_sents)',title='Avg #Sentences/Line')
            )

    return c

@st.cache_data    
def get_avg_feature_per_text_chart(df, datasetname, feature, feature_title, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length < max_textsize)]

    average_feature = 'mean('+feature+'):Q'

    if datasetname == 'All Datasets':
        c = alt.Chart(df_chart, width=800).mark_line(point=alt.OverlayMarkDef(filled=False, fill="white"), interpolate="monotone").encode(
            x=alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=30)),
            y=alt.Y(average_feature, title=feature_title),
            color="source:N"
        )        

        c = alt.Chart(df_chart, width=900, height=500).mark_bar(size=8).encode(
            x=alt.X('source:N',title=None, axis=alt.Axis(labels=False)),
            y=alt.Y(average_feature, title=feature_title),
            color=alt.Color('source:N',title=None, 
                            scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                            range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
                            ),
            column=alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=20), spacing = 20),
        ).properties(width=alt.Step(8))        


    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)

            c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_area(interpolate='monotone').encode(
                alt.X("length:Q", title='text length', 
                      axis=alt.Axis(domain=False, tickSize=0), 
                ),
                y=alt.Y(average_feature, title=feature_title, stack='center', axis=None),
                color=alt.Color('emotion:N', scale=emotion_scale)               
            )

            c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_bar(
               ).encode(
               alt.X("length:Q", title='Text size', bin=alt.Bin(maxbins=30)),
               y=alt.Y(average_feature, title=feature_title),
               color=alt.Color('emotion:N', scale=emotion_scale)
            )        


        else:
            c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text size', bin=alt.Bin(maxbins=30)),
                y=alt.Y(average_feature, title=feature_title)
            )

    return c

@st.cache_data
def get_correlation_matrix(df, col_labels_var1, col_labels_var2):
    df_aux = df
    df_aux.fillna('none', inplace=True)
    df_aux.reset_index(inplace=True, drop=True)

    if sorted(col_labels_var1) == sorted(col_labels_var2):          
        data = df_aux[col_labels_var1]
        corrMatrix = data.corr().reset_index().melt('index')
        corrMatrix.columns = ['var1', 'var2', 'correlation']
        
    else:
        data = df_aux[col_labels_var1 + col_labels_var2]
        corrMatrix = data.corr().reset_index().melt('index')
        corrMatrix.columns = ['var1', 'var2', 'correlation']

    return corrMatrix
    

@st.cache_data
def get_correlation_heatmap(dset_option, df, heatmap_relation, heatmap_title = 'Heatmap', min_value = None, max_value = None):
    if dset_option == 'GoEmotions':
        var1 = ge_labels
        if heatmap_relation == 'Emotions + emotions':
            var2 = ge_labels
    elif dset_option == 'CARER':
        var1 = carer_labels
    elif dset_option == 'ISEAR':
        var1 = isear_labels
    elif dset_option == 'TEC':
        var1 = tec_labels
    elif dset_option == 'SemEval2019 Task3':
        var1 = se_labels

    if heatmap_relation == 'Emotions + emotions (NRC)':
        var2 = nrc_labels
    elif heatmap_relation == 'Emotions + emotions (LIWC)':
        var2 = liwc_emotion_labels
    elif heatmap_relation == 'Emotions + tags (LIWC)':
        var2 = liwc_other_labels

    df = df[var1+var2]
    corrMatrix = get_correlation_matrix(df, var1, var2)
    if min_value:
        z_min = min_value
    if max_value:
        z_max = max_value

    chart = go.Figure(data=go.Heatmap(
            x=corrMatrix.var2,
            y=corrMatrix.var1,
            z=corrMatrix.correlation,
            type = 'heatmap',
            colorscale = 'RdBu',
            zmin=z_min,
            zmax=z_max
        )
    ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title='Pearson Correlation', autosize=False, width=1000, height=1000, margin_pad=10)

    return (chart), df


@st.cache_data
def get_correlation_heatmap_old(dset_option, df, var1, var2, var2_label = 'NRC', heatmap_title = 'Heatmap', min_value = None, max_value = None):
    if dset_option == 'GoEmotions':
        if var1 == 'emotion':
            var1 = ge_labels

        label_var1 = var1
        label_var2 = var2
        if var2 == 'emotion':
            var2 = ge_labels
        elif var2 == 'emotion_nrc':
            var2 = nrc_labels

        corrMatrix = get_correlation_matrix(df, var1, var2)
        if min_value:
            z_min = min_value
        if max_value:
            z_max = max_value

        chart = go.Figure(data=go.Heatmap(
                z=corrMatrix.correlation,
                x=corrMatrix.var2,
                y=corrMatrix.var1,
                type = 'heatmap',
                colorscale = 'RdBu',
                zmin=z_min,
                zmax=z_max
            )
        ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title='Pearson Correlation', autosize=False, width=1000, height=1000, margin_pad=10)

        return (chart), df

    else:
        corrMatrix = get_correlation_matrix(df, var1, var2)
        if min_value:
            z_min = min_value
        if max_value:
            z_max = max_value

        chart = go.Figure(data=go.Heatmap(
                z=corrMatrix.correlation,
                x=corrMatrix.var2,
                y=corrMatrix.var1,
                type = 'heatmap',
                colorscale = 'RdBu',
                zmin=z_min,
                zmax=z_max
            )
        ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title='Pearson Correlation', autosize=True, margin_pad=10) #, width=600, height=600)

        return (chart), df
    

@st.cache_data
def get_jaccard_dice_charts(df, var1, var2, z_min = 0, z_max = 1, chart_title = '', chart_width = 800, chart_height = 800):
    df_jd_chart = df

    liwc_emotions = []
    liwc_others = []

    if var2 == liwc_emotion_labels:
        liwc_emotions = [label.replace('liwc_','') for label in liwc_emotion_labels]
        var2_list = liwc_emotions
    elif var2 == liwc_other_labels:
        liwc_others = [label.replace('liwc_','') for label in liwc_other_labels]
        var2_list = liwc_others

    if isinstance(var2, list) and ( sorted(var2) == sorted(liwc_emotion_labels) or sorted(var2) == sorted(liwc_other_labels) ):
        df_jd_chart = df_jd_chart[(df_jd_chart.liwc.isin(var2_list)) & (df_jd_chart[var1].notnull()) & (df_jd_chart[var1] != '')]

    else:
        df_jd_chart = df_jd_chart[(df_jd_chart[var1].notnull()) & (df_jd_chart[var1] != '') & (df_jd_chart[var2].notnull()) & (df_jd_chart[var2] != '')]
    
    if var2 == liwc_emotion_labels:
        var2_t = 'emotion (LIWC)'
        var2_aux = 'liwc'
    elif var2 == liwc_other_labels:
        var2_t = 'other (LIWC)'
        var2_aux = 'liwc'
    else:
        var2_t = var2
        var2_aux = var2

    if chart_title.strip() == '':
        jaccard_title = var1+' X '+var2_t+' (Jaccard)'
        dice_title = var1+' X '+var2_t+' (Dice)'
    else:
        jaccard_title = chart_title+' (Jaccard)'
        dice_title = chart_title+' (Dice)'

    chart_jaccard = go.Figure(data=go.Heatmap(
                    z=df_jd_chart[var1+'_'+var2_aux+'_jaccard'],
                    x=df_jd_chart[var2_aux],
                    y=df_jd_chart[var1],
                    type = 'heatmap',
                    colorscale = 'Reds',
                    zmin=z_min,
                    zmax=z_max
                )
            ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title=jaccard_title, autosize=False, width=chart_width, height=chart_height)

    chart_jaccard = go.Figure(data=go.Heatmap(
                    z=df_jd_chart[var1+'_'+var2_aux+'_jaccard'],
                    x=df_jd_chart[var2_aux],
                    y=df_jd_chart[var1],
                    type = 'heatmap',
                    colorscale = 'Reds',
                    zmin=z_min,
                    zmax=z_max
                )
            ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title=jaccard_title, autosize=False, width=chart_width, height=chart_height)


    chart_dice = go.Figure(data=go.Heatmap(
                    z=df_jd_chart[var1+'_'+var2_aux+'_dice'],
                    x=df_jd_chart[var2_aux],
                    y=df_jd_chart[var1],
                    type = 'heatmap',
                    colorscale = 'Reds',
                    zmin=z_min,
                    zmax=z_max
                )
            ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title=dice_title, autosize=False, width=chart_width, height=chart_height)
    
    if df_jd_chart.empty:
        df_jd_chart = pd.DataFrame([var1], columns=['var1'])

    return chart_jaccard, chart_dice, df_jd_chart



@st.cache_data
def get_liwc_chart(df):
    chart = (
        alt.Chart(df, title="Emotions + LIWC Tags", width=1200, height=1200).mark_rect().encode(
            x=alt.X('emotion:N',title='emotion'),
            y=alt.Y('liwc_label:N',title='LIWC'),
            color=alt.Color('count:Q',title='#lines',scale=alt.Scale(scheme='Spectral')),
            tooltip = [
                alt.Tooltip('emotion:N',title="emotion"),
                alt.Tooltip('liwc_label:N'),
                alt.Tooltip('count:Q',title="#lines")]
        )
    )
    return (chart)




@st.cache_data
def get_dependency_treemap(dataset_name, emotion, min_counts):

    max_path_nodes = get_max_path_nodes(dataset_name, emotion, min_counts)

    level_columns = []
    for i in range(max_path_nodes):
        level_columns.append('level'+str(i))

    df_plot = get_path_levels(dataset_name, emotion, min_counts)

    df_plot = df_plot[['source', 'emotion', 'path', 'counts']+level_columns]
    
    df_plot.drop_duplicates(inplace=True)

    df_plot = df_plot.fillna('none')
    df_plot = df_plot.replace('','none')

    df_plot = df_plot.groupby(['source', 'emotion', 'path']+level_columns)['counts'].sum().reset_index(name='counts')

    fig = px.treemap(df_plot, path=['source','emotion']+level_columns, names='path', values='counts', 
                     color_continuous_scale='Peach', hover_data=['emotion','path','counts'])

    # Hide rectangles with "none"
    figure_data = fig["data"][0]
    mask = np.char.find(figure_data.ids.astype(str), "none") == -1
    figure_data.ids = figure_data.ids[mask]
    figure_data.values = figure_data.values[mask]
    figure_data.labels = figure_data.labels[mask]
    figure_data.parents = figure_data.parents[mask]

    fig.update_traces(tiling_packing="binary", selector=dict(type='treemap'))
    fig.update_layout(treemapcolorway = ["#f5ddb1","#fee1aa","#ff8c8c"], margin = dict(t=25, l=25, r=25, b=25))

    return fig



@st.cache_data
def get_entities_chart(df, datasetname):
    if datasetname == 'All Datasets':

        df['color'] = df['source'].apply(lambda x: get_dataset_color(x))
        #df['color'] = '#fee1aa'

        # create a list of unique nodes
        nodes = pd.concat([df['source'], df['ent_text']]).unique()

        # create a dictionary to map nodes to indices
        node_dict = {node: i for i, node in enumerate(nodes)}

        # create lists for source, target, and value
        sources = []
        targets = []
        values = []
        colors = []

        # loop through the dataframe and append values to the lists
        for i in range(len(df)):
            source_index = node_dict[df.iloc[i]['source']]
            target_index = node_dict[df.iloc[i]['ent_text']]
            sources.append(source_index)
            targets.append(target_index)
            colors.append(df.iloc[i]['color'])
            values.append(df.iloc[i]['count'])    

        # create the sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 10,
            thickness = 10,
            line = dict(color = "black", width = 0.5),
            label = nodes,
            color = "black"
            ),
            link = dict(
            source = sources,
            target = targets,
            value = values,
            color = colors,
            ))
        ])

        # display the chart
        fig.update_layout(title_text="Top 10 Entities/Dataset", 
                          font_size=14, 
                          font_family="Tahoma",
                          font_color="black",
                          width=1000, 
                          height=1000)

    else:

        df = df[df.source == datasetname]

        df['color'] = df['emotion'].apply(lambda x: get_emotion_color(x))
        #df['color'] = '#fee1aa'

        # create a list of unique nodes
        nodes = pd.concat([df['source'], df['ent_text'], df['emotion']]).unique()

        # create a dictionary to map nodes to indices
        node_dict = {node: i for i, node in enumerate(nodes)}

        # create lists for source, target, and value
        sources = []
        targets = []
        values = []
        colors = []

        # loop through the dataframe and append values to the lists
        for i in range(len(df)):
            source_index = node_dict[df.iloc[i]['ent_text']]
            target_index = node_dict[df.iloc[i]['emotion']]
            sources.append(source_index)
            targets.append(target_index)
            colors.append(df.iloc[i]['color'])
            values.append(df.iloc[i]['count'])    

        # create the sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 10,
            thickness = 10,
            line = dict(color = "black", width = 0.5),
            label = nodes,
            color = "black"
            ),
            link = dict(
            source = sources,
            target = targets,
            value = values,
            color = colors,
            ))
        ])

        # display the chart
        if datasetname == 'GoEmotions':
            fig.update_layout(title_text=datasetname+" - Top 3 Entities/Emotion", 
                              font_size=14, 
                              font_family="Tahoma",
                              font_color="black",                           
                              width=1000, 
                              height=1200)
        else:
            fig.update_layout(title_text=datasetname+" - Top 3 Entities/Emotion", 
                              font_size=14, 
                              font_family="Tahoma",
                              font_color="black",                             
                              width=1000, 
                              height=800)

    return fig