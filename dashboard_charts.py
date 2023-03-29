
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
nrc_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
liwc_emotion_labels = ['affect','posemo','negemo','anx','anger','sad']
liwc_other_labels = ['negate','social','cogmech','percept','bio','relativ','relig','death']


@st.cache_data
def get_comments(feature_option):
    if feature_option == '#Texts':
        return ['#Texts is important to identify the most relevant sizes of texts in the corpus.']
    elif feature_option == '#Duplicated Texts':
        return ['#The duplicated texts can disturb the machine learning task.']
    elif feature_option == 'Top Words':
        return ['Not done yet.']
    elif feature_option == '#Entities':
        return ['Not done yet.']
    elif feature_option == 'Entities':
        return ['Not done yet.']    
    elif feature_option == 'Average #Sentences/Text':
        return ["The #Sentences/Text according to text size can help to identify which emotion classes are the most expressed by sentences' concatenations."]
    elif feature_option == 'Average #Words/Text':
        return ['The #Words/Text according to text size is important to know how many elements are used to express the emotion.']
    elif feature_option == 'Average TTR/Text':
        return ['TTR = the total number of unique words / the total number of words.','A high TTR indicates a high degree of lexical variation.']
    elif feature_option == 'Average MTLD/Text':
        return ['Measure of Textual Lexical Diversity (MTLD) tries to account for and avoid the influence of text length.','This measure is the average length of the longest consecutive sequences of words that maintain a TTR of at least min_ttr (0.72 by default).']
    elif feature_option == 'Avg. Flesch-Kincaid Grade Level/Text':
        return ['The "Flesch–Kincaid Grade Level Formula" presents a score as a U.S. grade level to measure the readability level of various books and texts.']
    elif feature_option == 'Avg. Flesch Reading Ease/Text':
        return ['The Flesch Reading Ease gives a text a score between 1 and 100, with 100 being the highest readability score.']
    else:
        return []
    

@st.cache_data
def get_distfreq_chart(df, datasetname, chart_width, chart_height):
    if datasetname == 'All Datasets':
        c = alt.Chart(df, title="Emotion Frequency Distribution", width=chart_width, height=chart_height).mark_bar().encode(
            x='counts:Q',
            y=alt.Y('emotion:N', title='Emotion'),
            color=alt.Color('source:N', 
                            scale=alt.Scale(
                                domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])),
            order=alt.Order('source:N', sort='ascending'),
            #row=alt.Row('source:N', title='Dataset'),
            tooltip = [
                #alt.Tooltip('source:N'),
                alt.Tooltip('emotion:N',title='Emotion'),
                alt.Tooltip('source:N',title='Dataset'),
                alt.Tooltip('counts:Q')]
        )

    else:
        chart_emotions = sorted(list(set(df.emotion.tolist())))
        chart_colors = get_emotions_colors(chart_emotions)
        emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)

        c = alt.Chart(df, title="Emotion Frequency Distribution", width=chart_width, height=chart_height).mark_bar().encode(
            x='counts:Q',
            y=alt.Y('emotion:N', title='Emotion'),
            #color=alt.Color('emotion:N',title='Emotion', scale=alt.Scale(scheme='category20')),
            color=alt.Color('emotion:N', scale=emotion_scale, title='Emotion'),
            #row=alt.Row('source:N', title='Dataset'),
            tooltip = [
                #alt.Tooltip('source:N'),
                alt.Tooltip('emotion:N',title='Emotion'),
                alt.Tooltip('counts:Q')]
        )
    return c

    
@st.cache_data
def get_texts_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length <= max_textsize)]

    if datasetname == 'All Datasets':    
        #c = alt.Chart(df_chart, title="#Texts Frequency", width=800, height=600).mark_bar(
        #    ).encode(
        #    alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
        #    #alt.X("length:Q", title='text length'), #, bin=alt.Bin(maxbins=30)),
        #    y=alt.Y('count()',title='#Texts'),
        #    color=alt.Color('source:N', scale=alt.Scale(scheme='category20'))
        #)
        c = alt.Chart(df_chart, title="Texts Frequency", width=800, height=600).mark_area(interpolate='monotone').encode(
            alt.X("length:Q", title='text length', 
                  axis=alt.Axis(domain=False, tickSize=0), bin=alt.Bin(maxbins=100)
                  ),
                  alt.Y('count():Q', stack='center', axis=None),
                  color=alt.Color('source:N', 
                                  scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                                  range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']))
        )

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]        
        if group_by_emotions:
            chart_emotions = list(set(df_chart.emotion.tolist()))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
            #df_chart['color'] = df_chart['emotion'].apply(get_color)           
            c = alt.Chart(df_chart, title="#Texts Frequency By Emotions", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), #, bin=alt.Bin(maxbins=30)),
                y=alt.Y('count()',title='#Texts'),
                color=alt.Color('emotion:N', scale=emotion_scale)
                #color=alt.Color('emotion:N', scale=alt.Scale(scheme='category20'))
                #color=alt.Color('emotion:N', 
                #                  scale=alt.Scale(domain=emotions_chart),
                #                                  range=emotions_colors)
                # color=alt.Color('color', legend=alt.Legend('emotion')), # , scale=None, legend='emotion'
                #tooltip=['length:Q','count()','emotion']
                #tooltip=[alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)), 'count()', 'emotion']
            )
        else:
            c = alt.Chart(df_chart, title="#Texts Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), #, bin=alt.Bin(maxbins=30)),
                y=alt.Y('count()',title='#Texts')
            )

    return c

@st.cache_data
def get_duplicated_texts_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length <= max_textsize)]
    if not group_by_emotions:
        del df_chart['emotion']

    if datasetname == 'All Datasets':
        str_n_records = str(len(df_chart))
        # c = alt.Chart(df_chart, title="#Duplicated Texts ("+str_n_records+")", width=800, height=600).mark_bar(
        #     ).encode(
        #     alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
        #     #alt.X("length:Q", title='text length'), #, bin=alt.Bin(maxbins=30)),
        #     y=alt.Y('count():Q',title='#Texts'),
        #     color=alt.Color('source:N', scale=alt.Scale(scheme='category20'))
        # )
        c = alt.Chart(df_chart, title="#Duplicated Texts ("+str_n_records+")", width=800, height=600).mark_area(interpolate='monotone').encode(
            alt.X("length:Q", title='text length', 
                  axis=alt.Axis(domain=False, tickSize=0), bin=alt.Bin(maxbins=100)
                  ),
            alt.Y('count():Q', stack='center', axis=None),
            color=alt.Color('source:N', 
                            scale=alt.Scale(domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                                            range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']))
        )

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        str_n_records = str(len(df_chart))
        if group_by_emotions:            
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
            c = alt.Chart(df_chart, title="#Duplicated Texts ("+str_n_records+") Frequency By Emotions", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), #, bin=alt.Bin(maxbins=30)),
                y=alt.Y('count():Q',title='#Texts'),
                #color=alt.Color('emotion:N', scale=alt.Scale(scheme='category20'))
                color=alt.Color('emotion:N', scale=emotion_scale)
            )
        else:
            c = alt.Chart(df_chart, title="#Duplicated Texts ("+str_n_records+") Frequency By Emotions", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), #, bin=alt.Bin(maxbins=30)),
                y=alt.Y('count():Q',title='#Texts')
            )

    return c, df_chart

@st.cache_data
def get_words_per_text_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length <= max_textsize)]

    if datasetname == 'All Datasets':
        c = alt.Chart(df_chart, title="#Words/Text Frequency", width=800, height=600).mark_bar(
            ).encode(
            alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
            #alt.X("length:Q", title='text length'), 
            y=alt.Y('average(n_words)',title='Avg Words/Text'),
            color=alt.Color('source:N', scale=alt.Scale(scheme='category20'))
        )

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
            c = alt.Chart(df_chart, title="#Words/Text Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), 
                y=alt.Y('average(n_words)',title='Avg Words/Text'),
                #color=alt.Color('emotion:N', scale=alt.Scale(scheme='category20'))
                color=alt.Color('emotion:N', scale=emotion_scale)
            )
        else:
            c = alt.Chart(df_chart, title="#Words/Text Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), 
                y=alt.Y('average(n_words)',title='Avg Words/Text')
            )

    return c

@st.cache_data
def get_sentences_per_text_chart(df, datasetname, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length <= max_textsize)]

    if datasetname == 'All Datasets':
        c = alt.Chart(df_chart, title="#Sentences/Text Frequency", width=800, height=600).mark_bar(
            ).encode(
            alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
            #alt.X("length:Q", title='text length'), 
            y=alt.Y('average(n_sents)',title='Avg Sentences/Text'),
            color=alt.Color('source:N', scale=alt.Scale(scheme='category20'))
        )

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)
            c = alt.Chart(df_chart, title="#Sentences/Text Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), 
                y=alt.Y('average(n_sents)',title='Avg Sentences/Text'),
                #color=alt.Color('emotion:N', scale=alt.Scale(scheme='category20'))
                color=alt.Color('emotion:N', scale=emotion_scale)
            )
        else:
            c = alt.Chart(df_chart, title="#Sentences/Text Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), 
                y=alt.Y('average(n_sents)',title='Avg Sentences/Text')
            )

    
    return c

@st.cache_data    
def get_avg_feature_per_text_chart(df, datasetname, feature, feature_title, min_textsize, max_textsize, group_by_emotions = False):
    df_chart = df
    df_chart = df_chart[(df_chart.length >= min_textsize) & (df_chart.length <= max_textsize)]

    average_feature = 'average('+feature+')'

    if datasetname == 'All Datasets':
        c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_bar(
            ).encode(
            alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
            #alt.X("length:Q", title='text length'), 
            y=alt.Y(average_feature, title=feature_title),
            color=alt.Color('source:N', scale=alt.Scale(scheme='category20'))
        )

    else:
        df_chart = df_chart[(df_chart.source == datasetname)]
        if group_by_emotions:
            chart_emotions = sorted(list(set(df_chart.emotion.tolist())))
            chart_colors = get_emotions_colors(chart_emotions)
            emotion_scale = alt.Scale(domain=list(chart_emotions), range=chart_colors)

            #c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_bar(
            #    ).encode(
            #    alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
            #    #alt.X("length:Q", title='text length'), 
            #    y=alt.Y(average_feature, title=feature_title),
            #    color=alt.Color('emotion:N', scale=alt.Scale(scheme='category20'))
            #)

            c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_area(interpolate='monotone').encode(
                alt.X("length:Q", title='text length', 
                      axis=alt.Axis(domain=False, tickSize=0), #bin=alt.Bin(maxbins=150)
                ),
                y=alt.Y(average_feature, title=feature_title, stack='center', axis=None),
                color=alt.Color('emotion:N', scale=emotion_scale)
                #color=alt.Color('emotion:N', scale=alt.Scale(scheme='category20'))
                #color=alt.Color('emotion:N', 
                #                scale=alt.Scale(
                #                    domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                #                    range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']))                
            )
        else:
            c = alt.Chart(df_chart, title=feature_title+" Frequency", width=800, height=600).mark_bar(
                ).encode(
                alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=30)),
                #alt.X("length:Q", title='text length'), 
                y=alt.Y(average_feature, title=feature_title)
            )

    return c

@st.cache_data
def get_correlation_matrix(df, col_labels_var1, col_labels_var2):
    df_aux = df
    df_aux.fillna('none', inplace=True)
    # correlacao convencional entre os mesmos conjuntos de colunas do dataset
    if isinstance(col_labels_var1, list) and isinstance(col_labels_var2, list) and sorted(col_labels_var1) == sorted(col_labels_var1):  
        data = df_aux[col_labels_var1]
        corrMatrix = data.corr().reset_index().melt('index')
        corrMatrix.columns = ['var1', 'var2', 'correlation']
        
        return corrMatrix
    else:
        if isinstance(col_labels_var1, str):
            new_col_labels_var1 = []
            labels1 = sorted(set(df_aux[col_labels_var1].tolist()))
            for label in labels1:
                new_col_labels_var1.append(label+'_1')
                df_aux[label+'_1'] = df_aux[col_labels_var1].apply( lambda x: 1 if x==label else 0 )
        elif isinstance(col_labels_var1, list):
            new_col_labels_var1 = []
            for label in col_labels_var1:
                new_col_labels_var1.append(label+'_1')
                df_aux[label+'_1'] = df_aux[label].apply( lambda x: 1 if x>0 else 0 )
                
        if isinstance(col_labels_var2, str):
            new_col_labels_var2 = []
            labels2 = sorted(set(df_aux[col_labels_var2].tolist()))            
            for label in labels2:
                new_col_labels_var2.append(label+'_2')
                df_aux[label+'_2'] = df_aux[col_labels_var2].apply( lambda x: 1 if x==label else 0 )
        elif isinstance(col_labels_var2, list):
            new_col_labels_var2 = []
            for label in col_labels_var2:
                new_col_labels_var2.append(label+'_2')
                df_aux[label+'_2'] = df_aux[label].apply( lambda x: 1 if x>0 else 0 )
        
        data = df_aux[new_col_labels_var1 + new_col_labels_var2]
        corrMatrix = data.corr().reset_index().melt('index')
        corrMatrix.columns = ['var1', 'var2', 'correlation']
        
        for num in ['1', '2']:
            for col in ['var1', 'var2']:
                for value in sorted(corrMatrix[col].unique()):
                    if num in col and num not in value:
                        index = corrMatrix[corrMatrix[col] == value].index
                        corrMatrix.drop(index, inplace=True)
                        
        corrMatrix['var1'] = corrMatrix['var1'].apply( lambda x: x.replace('_1','') )
        corrMatrix['var2'] = corrMatrix['var2'].apply( lambda x: x.replace('_2','') )
        
        return corrMatrix

@st.cache_data
def get_correlation_heatmap(dset_option, df, var1, var2, var2_label = 'NRC', heatmap_title = 'Heatmap', min_value = None, max_value = None):
    if dset_option == 'GoEmotions':
        if var1 == 'emotion':
            var1 = ge_labels
        #elif var1 == 'emotion_nrc':
        #    var1 = nrc_labels

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

        #data = df_ge[ge_labels]
        #corrMatrix = data.corr().reset_index().melt('index')
        #corrMatrix.columns = ['var1', 'var2', 'correlation']

        #layout = go.Layout(
        #    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=label_var2)),
        #    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=label_var1)))

        chart = go.Figure(data=go.Heatmap(
                z=corrMatrix.correlation,
                x=corrMatrix.var2,
                y=corrMatrix.var1,
                type = 'heatmap',
                colorscale = 'Spectral',
                zmin=z_min,
                zmax=z_max
                #layout=layout
            )
        ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title='Correlação de Pearson', autosize=False, width=1000, height=1000, margin_pad=10)

        return (chart), df

        #chart = (
        #    alt.Chart(corrMatrix, title=heatmap_title, width=800, height=800).mark_rect().encode(
        #        x=alt.X('var2:N',title='emoção ('+var2_label+")"),
        #        y=alt.Y('var1:N',title='emoção'),
        #        color=alt.Color('correlation:Q',title='correlação',scale=alt.Scale(scheme='Spectral')),
        #        #color=alt.condition(selector, alt.Color('#lines:Q',title='#linhas',scale=alt.Scale(scheme='Spectral')), alt.value('black')),
        #        tooltip = [
        #            alt.Tooltip('var1:N'),
        #            alt.Tooltip('var2:N'),
        #            alt.Tooltip('correlation:Q')]
        #    )
        #)


        # base = alt.Chart(corrMatrix, title='heatmap_title').transform_filter(
        #     alt.datum.var1 < alt.datum.var2
        # ).encode(
        #     x=alt.X('var1',title=''),
        #     y=alt.Y('var2',title='')
        # ).properties(
        #     width=alt.Step(30),
        #     height=alt.Step(30)
        # )

        # rects = base.mark_rect().encode(
        #     color=alt.Color('correlation',scale=alt.Scale(scheme='Spectral'))
        # )

        # text = base.mark_text(
        #     size=10
        # ).encode(
        #     text=alt.Text('correlation', format=".2f"),
        #     color=alt.condition(
        #         "datum.correlation > 0.5",
        #         alt.value('white'),
        #         alt.value('black')
        #     ),
        #     tooltip = [
        #         alt.Tooltip('var1:N'),
        #         alt.Tooltip('var2:N'),
        #         alt.Tooltip('correlation:Q')]
        # )

        # chart = (
        #     rects + text
        # )

        return chart, df

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
                colorscale = 'Spectral',
                zmin=z_min,
                zmax=z_max
            )
        ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title='Correlação de Pearson', autosize=True, margin_pad=10) #, width=600, height=600)

        return (chart), df
    

@st.cache_data
def get_jaccard_dice_charts(df, var1, var2, z_min = 0, z_max = 1, chart_title = '', chart_width = 800, chart_height = 800):
    df_jd_chart = df.copy() 
    if isinstance(var2, list) and (var2 == liwc_emotion_labels or var2 == liwc_other_labels):
        df_jd_chart = df_jd_chart[(df_jd_chart.liwc.isin(var2)) & (df_jd_chart[var1].notnull())]
        #df_jd_chart = df_jd_chart[(df_jd_chart['liwc']=='affect') & (df_jd_chart[var1].notnull())]
        #df_jd_chart = df_jd_chart[(df_jd_chart[var1].notnull())]
    else:
        df_jd_chart = df_jd_chart[(df_jd_chart[var1].notnull()) & (df_jd_chart[var2].notnull())]
    
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
        jaccard_title = var1+' x '+var2_t+' (Jaccard)'
        dice_title = var1+' x '+var2_t+' (Dice)'
    else:
        jaccard_title = chart_title+' (Jaccard)'
        dice_title = chart_title+' (Dice)'

    chart_jaccard = go.Figure(data=go.Heatmap(
                    z=df_jd_chart[var1+'_'+var2_aux+'_jaccard'],
                    x=df_jd_chart[var2_aux],
                    y=df_jd_chart[var1],
                    type = 'heatmap',
                    colorscale = 'Spectral',
                    zmin=z_min,
                    zmax=z_max
                )
            ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title=jaccard_title, autosize=False, width=chart_width, height=chart_height)

    chart_dice = go.Figure(data=go.Heatmap(
                    z=df_jd_chart[var1+'_'+var2_aux+'_dice'],
                    x=df_jd_chart[var2_aux],
                    y=df_jd_chart[var1],
                    type = 'heatmap',
                    colorscale = 'Spectral',
                    zmin=z_min,
                    zmax=z_max
                )
            ).update_yaxes(autorange="reversed").update_xaxes(tickangle=-90).update_layout(title=dice_title, autosize=False, width=chart_width, height=chart_height)
    
    if df_jd_chart.empty:
        #df_jd_chart = df_jd_chart_aux
        df_jd_chart = pd.DataFrame([var1], columns=['var1'])

    return chart_jaccard, chart_dice, df_jd_chart



@st.cache_data
def get_liwc_chart(df):
    chart = (
        alt.Chart(df, title="Emotions x LIWC Tags", width=1200, height=1200).mark_rect().encode(
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
def get_morphological_treemap(df, dataset_name, emotion, max_path_nodes):
    level_columns = []
    for i in range(max_path_nodes):
        level_columns.append('level'+str(i))

    df_plot = df

    #common_emotions = ['anger', 'sadness', 'disgust', 'joy', 'shame', 'fear', 'guilt', 'surprise']

    df_plot = df_plot[['source', 'emotion', 'path', 'counts']+level_columns]

    #df = df[ (df.source=='SemEval2019 Task3') & (df.counts > 2) ]
    #df = df[ (df.source=='GoEmotions') & (df.counts > 2) & (df.emotion.isin(common_emotions))]
    df_plot = df_plot[ (df_plot.source==dataset_name) & (df_plot.emotion==emotion) & (df_plot.counts > 2)]
    df_plot.drop_duplicates(inplace=True)

    # Remove level columns with nan values
    for level in range(max_path_nodes):
        unique_values = df_plot['level'+str(level)].unique()
        if len(unique_values) == 1 and (unique_values[0] == np.nan or math.isnan(unique_values[0])):
            del df_plot['level'+str(level)]
            level_columns.remove('level'+str(level))

    df_plot = df_plot.fillna('none')

    df_plot = df_plot.groupby(['source', 'emotion', 'path']+level_columns)['counts'].sum().reset_index(name='counts')

    #for level in level_columns:
    #    df_plot[level] = df_plot[level].apply(lambda x: x if(x != 'none') else None)
    #df_plot = df_plot.applymap(lambda x: x if x else "none")    


    #df_plot["all"] = "all" # in order to have a single root node


    fig = px.treemap(df_plot, path=['source','emotion']+level_columns, names='path', values='counts') #, #color='counts',
                    #hover_data=['emotion','counts']) #, color_continuous_scale='Sunset') #, branchvalues="total")

    figure_data = fig["data"][0]
    mask = np.char.find(figure_data.ids.astype(str), "none") == -1
    figure_data.ids = figure_data.ids[mask]
    figure_data.values = figure_data.values[mask]
    figure_data.labels = figure_data.labels[mask]
    figure_data.parents = figure_data.parents[mask]

    #fig.update_traces(root_color="lightgrey")
    fig.update_layout(treemapcolorway = [get_color(emotion)], margin = dict(t=25, l=25, r=25, b=25))

    return fig

