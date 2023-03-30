import altair as alt
import pandas as pd
import streamlit as st
import time
#import locale
import base64
import subprocess
import sys
from os.path import exists

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import zipfile
except:
    install("zipfile")
    import zipfile

if not exists("./data/goemotions_no_text_stats.csv") and exists("./data/goemotions_no_text_stats.zip"):
    with zipfile.ZipFile('./data/goemotions_no_text_stats.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

if not exists("./data/source_emotion_path_levels.csv") and exists("./data/source_emotion_path_levels.zip"):
    with zipfile.ZipFile('./data/source_emotion_path_levels.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

#install("streamlit-aggrid")
#install("plotly")

#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
#locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# import panel as pn
# pn.extension('vega')

# É necessário instalar a biblioteca streamlit-aggrid, conforme abaixo
# pip install streamlit-aggrid
try:
    from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
    #from streamlit_option_menu import option_menu
except:
    install("streamlit-aggrid")
    from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

import streamlit.components.v1 as components

try:
    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go
except:
    install("plotly")
    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go


#import plotly.plotly as py
from plotly.graph_objs import *
#py.sign_in('username', 'api_key')

pio.templates.default = "simple_white"

#from streamlit_plotly_events import plotly_events

from dashboard_charts import *
from dashboard_data import *

from PIL import Image
logo_image = Image.open('./data/logo64.png')

alt.themes.enable("streamlit")

APP_TITLE = "Vis-EMotif"

st.set_page_config(
    page_title=APP_TITLE, page_icon=logo_image, layout='wide' #layout="centered" or "wide"
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1.5rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

DATA_DIR = './data'
dataset_dir = DATA_DIR

GO_EMOTIONS_DIR = dataset_dir
HF_DIR = dataset_dir
ISEAR_DIR = dataset_dir
SAIF_MOHAMMAD_DIR = dataset_dir
SEMEVAL19_DIR = dataset_dir

# Loading large datasets
##df_ge = pd.read_csv(GO_EMOTIONS_DIR+'/goemotions_no_text_stats.csv', encoding='UTF-8', sep='|')

#df_ge = pd.read_csv(GO_EMOTIONS_DIR+'/goemotions_nrc_stats.csv', encoding='UTF-8', sep='|')
#del df_ge['text']
#del df_ge['texto']
#df_ge.to_csv(dataset_dir+'/goemotions_no_text_stats.csv', sep='|', encoding='UTF-8', index=False)

##df_hf = pd.read_csv(HF_DIR+'/carer_no_text_stats.csv', encoding='UTF-8', sep='|')

#df_hf = pd.read_csv(HF_DIR+'/emotion_huggingface_nrc_stats.csv', encoding='UTF-8', sep='|')
#del df_hf['text']
#del df_hf['texto']
#df_hf.to_csv(dataset_dir+'/carer_no_text_stats.csv', sep='|', encoding='UTF-8', index=False)

##df_isear = pd.read_csv(ISEAR_DIR+'/isear_no_text_stats.csv', encoding='UTF-8', sep='|')

#df_isear = pd.read_csv(ISEAR_DIR+'/isear_nrc_stats.csv', encoding='UTF-8', sep='|')
#del df_isear['text']
#del df_isear['texto']
#df_isear.to_csv(dataset_dir+'/isear_no_text_stats.csv', sep='|', encoding='UTF-8', index=False)

##df_saif = pd.read_csv(SAIF_MOHAMMAD_DIR+'/tec_no_text_stats.csv', encoding='UTF-8', sep='|')

#df_saif = pd.read_csv(SAIF_MOHAMMAD_DIR+'/saifmohammad_nrc_stats.csv', encoding='UTF-8', sep='|')
#del df_saif['text']
#del df_saif['texto']
#df_saif.to_csv(dataset_dir+'/tec_no_text_stats.csv', sep='|', encoding='UTF-8', index=False)

##df_se = pd.read_csv(SEMEVAL19_DIR+'/se_no_text_stats.csv', encoding='UTF-8', sep='|')

#df_se = pd.read_csv(SEMEVAL19_DIR+'/se19_nrc_stats.csv', encoding='UTF-8', sep='|')
#del df_se['text']
#del df_se['texto']
#df_se.to_csv(dataset_dir+'/se_no_text_stats.csv', sep='|', encoding='UTF-8', index=False)

#df_source = pd.read_csv(dataset_dir+'/data_source.csv', encoding='UTF-8', sep='|')
#df_source = df_source[['source','counts']]

#df_source_emotion = pd.read_csv(dataset_dir+'/data_source_emotion.csv', encoding='UTF-8', sep='|')
#df_source_emotion = df_source_emotion[['source','emotion','counts']]

#df_emotion_source = df_source_emotion

#df_emotion_source = pd.read_csv(dataset_dir+'/data_text_emotion_source.csv', encoding='UTF-8', sep='|')
#df_emotion_source = df_emotion_source[['emotion','source','counts']]

#df_emotion_source_length = pd.read_csv(dataset_dir+'/emotion_source_length.csv', encoding='UTF-8', sep='|')
#df_emotion_source_length = pd.read_csv(dataset_dir+'/data_text_emotion_source_length.csv', encoding='UTF-8', sep='|')
#df_emotion_source_length = df_emotion_source_length[['emotion','source','length']]
#df_emotion_source_length.to_csv(dataset_dir+'/emotion_source_length.csv', sep='|', encoding='UTF-8', index=False)


#df_text_emotion_source_stats = pd.read_csv(dataset_dir+'/emotion_source_stats.csv', encoding='UTF-8', sep='|')
#df_text_emotion_source_stats = pd.read_csv(dataset_dir+'/data_text_emotion_source_stats.csv', encoding='UTF-8', sep='|')

#df_text_emotion_source_dup = df_text_emotion_source_stats.copy()
#df_text_emotion_source_dup = df_text_emotion_source_dup[['text','emotion','source']]
#df_text_emotion_source_dup = df_text_emotion_source_dup.groupby(['text','emotion','source']).size().reset_index(name='counts')
#df_text_emotion_source_dup.to_csv(dataset_dir+'/data_text_emotion_source_dup.csv', sep='|', encoding='UTF-8', index=False)

#df_ge_dup = df_ge.copy()
#df_ge_dup = df_ge_dup.groupby(['text']).size().reset_index(name='counts')
#df_ge_dup = df_ge_dup[df_ge_dup.counts > 1]
#df_ge_dup.to_csv(dataset_dir+'/data_dup_ge.csv', sep='|', encoding='UTF-8', index=False)

#st.write(len(df_ge_dup))

df_text_emotion_source_dup = pd.read_csv(dataset_dir+'/data_text_emotion_source_dup.csv', encoding='UTF-8', sep='|')
#df_text_emotion_source_dup = df_text_emotion_source_dup[df_text_emotion_source_dup.counts > 1]
#df_text_emotion_source_dup['length'] = df_text_emotion_source_dup['text'].apply(len)

#df_text_emotion_source_dup_ge = df_text_emotion_source_dup.copy()
#df_text_emotion_source_dup_ge = df_text_emotion_source_dup_ge[df_text_emotion_source_dup_ge.source == 'GoEmotions']
#df_text_emotion_source_dup_ge = pd.merge(df_text_emotion_source_dup_ge, df_ge_dup, how ='inner', on =['text'])
#st.write(len(df_text_emotion_source_dup_ge))
#df_text_emotion_source_dup_ge.to_csv(dataset_dir+'/data_text_emotion_source_dup_ge.csv', sep='|', encoding='UTF-8', index=False)

#df_text_source_dup = df_text_emotion_source_stats.copy()
#df_text_source_dup = df_text_source_dup[['text','source']]
#df_text_source_dup = df_text_source_dup.groupby(['text','source']).size().reset_index(name='counts')
#df_text_source_dup.to_csv(dataset_dir+'/data_text_source_dup.csv', sep='|', encoding='UTF-8', index=False)

#df_text_source_dup = pd.read_csv(dataset_dir+'/data_text_source_dup.csv', encoding='UTF-8', sep='|')

#df_text_source_dup = df_text_source_dup[df_text_source_dup.counts > 1]
#df_text_source_dup['length'] = df_text_source_dup['text'].apply(len)

#df_text_source_dup_ge = df_text_source_dup.copy()
#df_text_source_dup_ge = df_text_source_dup_ge[df_text_source_dup_ge.source == 'GoEmotions']
#df_text_source_dup_ge = pd.merge(df_text_source_dup_ge, df_ge_dup, how ='inner', on =['text'])
#st.write(len(df_text_source_dup_ge))
#df_text_emotion_source_dup_ge.to_csv(dataset_dir+'/data_text_source_dup_ge.csv', sep='|', encoding='UTF-8', index=False)

#df_path_levels = pd.read_csv(dataset_dir+'/source_emotion_path_levels.csv', encoding='UTF-8', sep='|')
#df_path_levels = pd.read_csv(dataset_dir+'/data_source_emotion_text_sentence_path_levels.csv', encoding='UTF-8', sep='|')
#del df_path_levels['text'] 
#del df_path_levels['sentence'] 
#df_path_levels.to_csv(dataset_dir+'/source_emotion_path_levels.csv', sep='|', encoding='UTF-8', index=False)
#max_path_nodes = df_path_levels.path_length.max()

#del df_text_emotion_source_stats['text']
#df_text_emotion_source_stats.to_csv(dataset_dir+'/emotion_source_stats.csv', sep='|', encoding='UTF-8', index=False)

#ge_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Jaccard/Dice

#df_ge_jd = pd.read_csv(dataset_dir+'/goemotions_jd.csv', encoding='UTF-8', sep='|')
#df_hf_jd = pd.read_csv(dataset_dir+'/emotion_huggingface_jd.csv', encoding='UTF-8', sep='|')
#df_isear_jd = pd.read_csv(dataset_dir+'/isear_jd.csv', encoding='UTF-8', sep='|')
#df_saif_jd = pd.read_csv(dataset_dir+'/saif_jd.csv', encoding='UTF-8', sep='|')
#df_se_jd = pd.read_csv(dataset_dir+'/se_jd.csv', encoding='UTF-8', sep='|')


#df_ge_not_bin = pd.DataFrame(columns=['text','emotion'])
#for label in ge_labels:
#    df_ge_aux = df_ge[df_ge[label] == 1]
#    df_ge_aux['emotion'] = label
#    df_ge_not_bin = pd.concat([df_ge_not_bin, df_ge_aux[['text','emotion']]])

# New Summary DataFrame
#df_emotion_summary = pd.DataFrame(columns=['text','emotion','source'])

#df_ge['source'] = 'GoEmotions'
#df_ge_not_bin['source'] = 'GoEmotions'


#df_emotion_summary = pd.concat([df_emotion_summary, df_ge_not_bin[['text','emotion','source']]])
#df_hf['source'] = 'CARER'
#df_emotion_summary = pd.concat([df_emotion_summary, df_hf[['text','emotion','source']]])
#df_isear['source'] = 'ISEAR'
#df_emotion_summary = pd.concat([df_emotion_summary, df_isear[['text','emotion','source']]])
#df_saif['source'] = 'TEC'
#df_emotion_summary = pd.concat([df_emotion_summary, df_saif[['text','emotion','source']]])
#df_se['source'] = 'SemEval2019 Task3'
#df_emotion_summary = pd.concat([df_emotion_summary, df_se[['text','emotion','source']]])

#df_emotion_summary['emotion'] = df_emotion_summary['emotion'].replace(['angry', 'happy', 'sad'], ['anger', 'joy', 'sadness'])

#df_emotion_summary.to_csv(dataset_dir+'/data_text_emotion_source.csv', sep='|', encoding='UTF-8')

#df_emotion_summary_text = df_emotion_summary.copy()
#df_emotion_summary_text['length'] = df_emotion_summary_text['text'].apply(len)

#df_emotion_summary_text.to_csv(dataset_dir+'/data_text_emotion_source_length.csv', sep='|', encoding='UTF-8')

#df_source_size = pd.concat([
#    df_ge[['text','source']],
#    df_hf[['text','source']],
#    df_isear[['text','source']],
#    df_saif[['text','source']],
#    df_se[['text','source']] ])

#df_source_size = df_source_size.groupby(['source']).size().reset_index(name='counts')
#df_source_size.to_csv(dataset_dir+'/data_source.csv', sep='|', encoding='UTF-8')

#source_data_frame = df_emotion_summary.groupby(['source','emotion']).size().reset_index(name='counts')
#source_data_frame.to_csv(dataset_dir+'/data_source_emotion.csv', sep='|', encoding='UTF-8')

@st.cache_data
def get_current_dataset(dataset_option):
    return get_df_stats(dataset_option)
    #if dataset_option == 'CARER':
    #    return df_hf
    #elif dataset_option == 'GoEmotions':
    #    return df_ge
    #elif dataset_option == 'ISEAR':
    #    return df_isear
    #elif dataset_option == 'TEC':
    #    return df_saif
    #elif dataset_option == 'SemEval2019 Task3':
    #    return df_se
    #else:
    #    return pd.DataFrame()
    
@st.cache_data    
def get_emotions_labels(dataset_option):
    df_emotion_source_ = get_source_emotion()
    if dataset_option != 'All Datasets':
        df_emotion_source_ = df_emotion_source_[df_emotion_source_.source == dataset_option]

    # Filter dataset
    emo_labels = list(df_emotion_source_['emotion'].unique())
    return emo_labels


#source_data_frame = df_emotion_source
#data_frame = source_data_frame


# Dashboard Title
#st.markdown("#### "+APP_TITLE+"")

# Main Dashboard Tabs
tab_home, tab_distfreq, tab_stats, tab_variables_rel, tab_morphotree = st.tabs(["Home", "Emotion Distribution", "Corpora Statistics", "Variables and Relationships", "Morphological Trees"])

# Home Tab
with tab_home:
    #col_logo, col_title, col_empty = st.columns([1,11,1])
    #with col_logo:
    #    st.image(logo_image)
    #with col_title:

    LOGO_IMAGE = "./data/logo64.png"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:20px !important;
            color: #000000 !important;
            padding-top: 16px !important;
            padding-left: 10px !important;
        }
        .logo-img {
            float:right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">Vis-EMotif - Visual Analysis of Textual Emotion Detection Datasets</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    #st.markdown("##### EMotif - Visual Analysis of Textual Emotion Detection Datasets")
    st.markdown("")
    st.markdown("This dashboard compares **five** benchmark datasets (corpora) used for textual emotion detection. Beyond statistics and linguistic features to be verified, there are other issues to examine: **imbalanced data**, **variable interference** and **data diversity/complexity**.")

    col_overview_table, col_overview_chart = st.columns([1,1])
    with col_overview_table:
        df_datasets_overview = get_datasets_overview()
        st.dataframe(df_datasets_overview)

        # df_overview = get_datasets_overview()

        # gb1 = GridOptionsBuilder.from_dataframe(df_overview)

        # gb1.configure_column("Year", headerName="Year", width=80)
        # gb1.configure_column("#Emotions", headerName="#Emotions", width=110)
        # gb1.configure_pagination(paginationAutoPageSize=False)
        # gb1.configure_grid_options(domLayout='normal')
        
        # gridOptions1 = gb1.build()
   
        # grid_response1 = AgGrid(
        #     df_overview, 
        #     gridOptions=gridOptions1,
        #     height=300, 
        #     width='100%',
        #     theme='streamlit',
        #     fit_columns_on_grid_load=True,
        #     allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        #     enable_enterprise_modules=False,
        # )

        #with open('datasets.html') as f:
        #    data = f.read()
        #    components.html(data, width=700, height=450)

    with col_overview_chart:
        with st.spinner("Preparing the datasets' overview chart..."):
            data_frame_count = get_df_all_source_counts()
            c_count = alt.Chart(data_frame_count, title="Datasets Sizes", width=600, height=250).mark_bar().encode(
                x=alt.X('counts:Q', title='#Texts'),
                y=alt.Y('source:N', title='Dataset'),
                color=alt.Color('source:N',title='Dataset',legend=None, scale=alt.Scale(scheme='category20')),
                tooltip = [
                    alt.Tooltip('source:N',title='Dataset'),
                    alt.Tooltip('counts:Q')]
            )
            st.altair_chart(c_count, use_container_width=True)

    col_left, col_center, col_right = st.columns([1,10,1])
    with col_center:
        st.markdown('##### Guidelines');
        st.markdown('**"Emotion Distribution"** presents the balance of the data according to emotion classes. **"Corpora Statistics"** shows linguistical and descriptive metrics to explore the datasets. '+
                    'The **"Variables and Relationships"** tab helps to detect interference between variables. The **last one tab** presents morphological treemaps to examine the vocabulary complexity of the corpora.')



# Emotion Frequency Distribution
with tab_distfreq:

    col_dataset, col_chart = st.columns([1,5])

    with col_dataset:

        dataset_selected = st.selectbox(
            label='Choose Corpus', 
            key='Dataset', 
            options=['All Datasets','CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
            label_visibility="visible"
        )

        emotion_filtered = st.selectbox(
            label='Filter Emotions', 
            key='FilterEmotions', 
            options=['Only Primary Emotions','All Except Neutral and Others','All'],
            label_visibility="visible"
        )

        # Filter dataset
        all_emotion_labels = get_emotions_labels(dataset_selected)
        all_emotion_labels_default = get_emotions_labels(dataset_selected)
        if emotion_filtered == 'All Except Neutral and Others':
            try:
                all_emotion_labels_default.remove('neutral')
            except:
                pass            
            try:
                all_emotion_labels_default.remove('others')
            except:
                pass
        elif emotion_filtered == 'Only Primary Emotions':
            all_emotion_labels_default = [e for e in all_emotion_labels if e in ['anger','disgust','fear','joy','sadness','surprise']]

        # Filter dataset

        emotions_selected = st.multiselect(
            label='Choose Individual Emotions',
            options=all_emotion_labels,
            default=all_emotion_labels_default
        )

        with st.form("emotionsf_form"):

            df_emotion_histogram = get_source_emotion_length()
            max_length2 = int(df_emotion_histogram['length'].max())
            max_textsize2 = st.slider(key = 'slider2', label = 'Maximum Text Size', min_value = 0, max_value = max_length2, value = max_length2, step=10)

            emotionsfreq_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)

    with col_chart:
        #st.markdown('Examining the balance of the data can help to improve the corpus and to minimize bias in the models resulting from these data.')            

        if emotionsfreq_button_submited:

            with st.spinner("Preparing the chart..."):
                emotions_removed = [x for x in all_emotion_labels if x not in emotions_selected]

                if max_textsize2 != max_length2:
                    source_data_frame2 = get_source_emotion_length()
                    source_data_frame2 = source_data_frame2[source_data_frame2.length <= max_textsize2]
                    source_data_frame2 = source_data_frame2.groupby(['source','emotion']).size().reset_index(name='counts')
                    data_frame = source_data_frame2
                else:
                    source_data_frame = get_source_emotion()
                    data_frame = source_data_frame

                # Filter emotions
                for emotion in emotions_removed:
                    data_frame = data_frame[ data_frame["emotion"] != emotion ]

                # Filter dataset
                if dataset_selected != 'All Datasets':
                    data_frame = data_frame[ data_frame["source"] == dataset_selected ]

                if len(data_frame) <= 30:
                    chart_height = 500
                else:
                    chart_height = 1000

                c = get_distfreq_chart(data_frame, dataset_selected, 800, chart_height)

                # if dataset_selected == 'All Datasets':
                #     c = alt.Chart(data_frame, title="Emotion Frequency Distribution", width=800, height=chart_height).mark_bar().encode(
                #         x='counts:Q',
                #         y=alt.Y('emotion:N', title='Emotion'),
                #         #color=alt.Color('source:N', title='Dataset', scale=alt.Scale(scheme='category20')),
                #         color=alt.Color('source:N', 
                #                         scale=alt.Scale(
                #                             domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                #                             range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])),
                #         order=alt.Order('source:N', sort='ascending'),
                #         #row=alt.Row('source:N', title='Dataset'),
                #         tooltip = [
                #             #alt.Tooltip('source:N'),
                #             alt.Tooltip('emotion:N',title='Emotion'),
                #             alt.Tooltip('source:N',title='Dataset'),
                #             alt.Tooltip('counts:Q')]
                #     )

                # else:
                #     c = alt.Chart(data_frame, title="Emotion Frequency Distribution", width=800, height=chart_height).mark_bar().encode(
                #         x='counts:Q',
                #         y=alt.Y('emotion:N', title='Emotion'),
                #         color=alt.Color('emotion:N',title='Emotion', scale=alt.Scale(scheme='category20')),
                #         #row=alt.Row('source:N', title='Dataset'),
                #         tooltip = [
                #             #alt.Tooltip('source:N'),
                #             alt.Tooltip('emotion:N',title='Emotion'),
                #             alt.Tooltip('counts:Q')]
                #     )

                st.session_state.distfreq_chart = c
                st.altair_chart(c, use_container_width=True)

        else:
            if 'distfreq_chart' in st.session_state:
                c = st.session_state.distfreq_chart
                st.altair_chart(c, use_container_width=True)
                

                # if dataset_selected == 'All Datasets':
                #     # Display text length chart for each dataset with color according to dataset
                #     df_emotion_histogram = df_emotion_summary.copy()
                #     df_emotion_histogram['length'] = df_emotion_histogram['text'].apply(len)
                #     for emotion in emotions_removed:
                #         df_emotion_histogram = df_emotion_histogram[ df_emotion_histogram["emotion"] != emotion ]

                #     base = alt.Chart(df_emotion_histogram).properties(title="Text Length by Emotion", width=900)

                #     line = base.mark_area(opacity=0.5, interpolate='monotone', point=True).encode(
                #         alt.X("length:Q", title='text length', bin=alt.Bin(maxbins=40)),
                #         y='count()',
                #         color=alt.Color('source:N'),
                #         row=alt.Order('emotion:N', sort='ascending')
                #     )
                    
                #     h = line

                #     st.altair_chart(h, use_container_width=False)

                # if dataset_selected == 'All Datasets':
                #     df_emotion_histogram = df_emotion_histogram[df_emotion_histogram.length <= max_textsize2]

                #     h = alt.Chart(df_emotion_histogram, title="Text Length Frequency", width=800, height=600).mark_area(interpolate='monotone').encode(
                #         alt.X("length:Q", title='text length', 
                #             axis=alt.Axis(domain=False, tickSize=0), bin=alt.Bin(maxbins=100)
                #         ),
                #         alt.Y('count():Q', stack='center', axis=None),
                #         color=alt.Color('source:N', 
                #                         scale=alt.Scale(
                #                             domain=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
                #                             range=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']))
                #     )

                #     st.altair_chart(h, use_container_width=True)
                # else:
                #     # Instead of displaying text length chart for each dataset, define color according to emotion
                #     df_emotion_histogram = df_emotion_source_length

                #     df_emotion_histogram = df_emotion_histogram[df_emotion_histogram.length <= max_textsize2]

                #     df_emotion_histogram = df_emotion_histogram[df_emotion_histogram.source == dataset_selected]
                #     for emotion in emotions_removed:
                #         df_emotion_histogram = df_emotion_histogram[ df_emotion_histogram["emotion"] != emotion ]

                #     base = alt.Chart(df_emotion_histogram).properties(title="Text Length by Emotion", width=900)

                #     h = alt.Chart(df_emotion_histogram, title="Text Length by Emotion", width=800, height=600).mark_area(interpolate='monotone').encode(
                #         alt.X("length:Q", title='text length', 
                #             axis=alt.Axis(domain=False, tickSize=0), bin=alt.Bin(maxbins=100)
                #         ),
                #         alt.Y('count():Q', stack='center', axis=None),
                #         alt.Color('emotion:N',
                #             scale=alt.Scale(scheme='category20')
                #         )
                #     )

                #     st.altair_chart(h, use_container_width=True)

                    # Olhar -> https://altair-viz.github.io/user_guide/compound_charts.html


# Statistics Tab
# https://textacy.readthedocs.io
with tab_stats:

    df_emotion_histogram = get_source_emotion_length()
    max_length = int(df_emotion_histogram['length'].max())
   
    col_filters, col_charts_stats = st.columns([1,5])
    with col_filters:
        dset_selected = st.selectbox(
            label='Choose Corpus', 
            key='Dset', 
            options=['All Datasets','CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
            label_visibility="visible"
        )

        if dset_selected == 'All Datasets':
            value_default_by_emotion = False
            by_emotion_disabled = True
        else:
            value_default_by_emotion = True
            by_emotion_disabled = False

        by_emotion = st.checkbox('Group By Emotions', value=value_default_by_emotion, key='ByEmotion', label_visibility="visible", disabled=by_emotion_disabled)

        feature_selected = st.selectbox(
            label='Feature', 
            key='Features', 
            options=['#Texts','#Duplicated Texts','Average #Sentences/Text','Average #Words/Text','Top Words','#Entities',
                    'Entities','Average TTR/Text','Avg. Flesch-Kincaid Grade Level/Text', # 'Average MTLD/Text'
                    'Avg. Flesch Reading Ease/Text'],
            help='Information about the feature will be below the chart',
            label_visibility="visible"
        )    

        with st.form("stats_form"):
            options_slider = [x for x in range(0, max_length+1, 10)]
            options_slider.append(max_length)
            min_textsize, max_textsize = st.select_slider(key = 'slider1', label = 'Text Size', value=(0, max_length), options=options_slider, label_visibility="visible")
            #max_textsize = st.slider(key = 'slider1', label = 'Maximum Text Size', min_value = 0, max_value = max_length, value = max_length, step=10)

            #reload_choices = st.checkbox('Reload Last Choices', value=True, key='ReloadChoices', label_visibility="visible")

            stats_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)        
        
        comments = get_comments(feature_selected)
        if len(comments)>0:
            st.markdown('**Tip**')
            for comment in comments:
                st.caption(comment)

    with col_charts_stats:
        #st.markdown('Visualizing corpus statistics can help to identify the most useful characteristics of the corpus and which transformations can increase its data quality.')
        if stats_button_submited:                
            #comments = get_comments(feature_selected)
            #if len(comments)>0:
            #    for comment in comments:
            #        st.caption(comment)
            with st.spinner("Preparing the chart..."):
                if feature_selected == '#Texts':
                    h = get_texts_chart(get_source_emotion_stats(), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                elif feature_selected == '#Duplicated Texts':
                    h, df_h = get_duplicated_texts_chart(df_text_emotion_source_dup, dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)

                    # gb2 = GridOptionsBuilder.from_dataframe(df_h)

                    # gb2.configure_pagination(paginationAutoPageSize=True)
                    # gb2.configure_grid_options(domLayout='normal')
                    # gridOptions2 = gb2.build()

                    # grid_response2 = AgGrid(
                    #     df_h, 
                    #     gridOptions=gridOptions2,
                    #     height=600, 
                    #     width='100%',
                    #     theme='streamlit',
                    #     fit_columns_on_grid_load=True,
                    #     allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                    #     enable_enterprise_modules=False,
                    # )

                elif feature_selected == 'Average #Sentences/Text':
                    h = get_sentences_per_text_chart(get_source_emotion_stats(), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                elif feature_selected == 'Average #Words/Text':
                    h = get_words_per_text_chart(get_source_emotion_stats(), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                elif feature_selected == 'Top Words':
                    h = None
                    if 'tab_chart' in st.session_state:
                        del st.session_state['tab_chart']
                    #st.altair_chart(h, use_container_width=True)
                elif feature_selected == '#Entities':
                    h = None
                    if 'tab_chart' in st.session_state:
                        del st.session_state['tab_chart']                    
                    #st.altair_chart(h, use_container_width=True)
                elif feature_selected == 'Entities':
                    h = None
                    if 'tab_chart' in st.session_state:
                        del st.session_state['tab_chart']                    
                    #st.altair_chart(h, use_container_width=True)                
                elif feature_selected == 'Average TTR/Text':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats(), dset_selected, 'diversity_ttr', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                elif feature_selected == 'Average MTLD/Text':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats(), dset_selected, 'diversity_mtld', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                elif feature_selected == 'Avg. Flesch-Kincaid Grade Level/Text':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats(), dset_selected, 'flesch_kincaid_grade_level', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                elif feature_selected == 'Avg. Flesch Reading Ease/Text':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats(), dset_selected, 'flesch_reading_ease', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=True)
                
                if h:
                    st.session_state.tab_chart = h


        else:
            if 'tab_chart' in st.session_state:
                h = st.session_state.tab_chart
                st.altair_chart(h, use_container_width=True)

            


with tab_variables_rel:

    col_filters_var, col_chart_var = st.columns([1,4])

    with col_filters_var:

        dset_selected_var = st.selectbox(
            label='Choose Corpus', 
            key='DatasetVar', 
            options=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
            label_visibility="visible"
        )

        heatmap_option = st.selectbox(
            label='Metric',
            key='HeatmapType',
            options=['Pearson','Jaccard','Dice'],
            label_visibility="visible")
        
        if heatmap_option == 'Pearson':
            h_min = -1
            h_max = 1
            h_values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        elif heatmap_option == 'Jaccard':
            h_min = 0
            h_max = 1
            h_values = [0, 0.25, 0.5, 0.75, 1]
        elif heatmap_option == 'Dice':
            h_min = 0
            h_max = 1
            h_values = [0, 0.25, 0.5, 0.75, 1]
        
        if heatmap_option == 'Pearson':
            # GoEmotions has different variables for each emotion class, the others have only one variable
            if dset_selected_var == 'GoEmotions':
                relations_available = ['Emotions X emotions','Emotions X emotions (NRC)','Emotions X emotions (LIWC)','Emotions X tags (LIWC)']
            else:
                relations_available = ['Emotions X emotions (NRC)','Emotions X emotions (LIWC)','Emotions X tags (LIWC)']    
        else:
            relations_available = ['Emotions X emotions (NRC)','Emotions X emotions (LIWC)','Emotions X tags (LIWC)']
        
        heatmap_relation = st.selectbox(
            label = 'Relationship',
            key = 'HeatmapRelation',
            options = relations_available,
            label_visibility = "visible")
        
        with st.form("var_form"):

            heatmap_min, heatmap_max = st.select_slider(
                key='min_max_values',
                label='Min - Max',
                options=h_values,
                value=(h_min, h_max),
                )
            
            #reload_choices3 = st.checkbox('Reload Last Choices', value=True, key='ReloadChoices3', label_visibility="visible")

            heatmap_vars = {
                'Dataset Emotions': 'emotion',
                'NRC Emotions': 'emotion_nrc',
                'LIWC Emotions': ['liwc_affect','liwc_posemo','liwc_negemo','liwc_anx','liwc_anger','liwc_sad'],
                'LIWC Tags': ['liwc_negate','liwc_social','liwc_cogmech','liwc_percept','liwc_bio','liwc_relativ','liwc_relig','liwc_death']
            }

            heatmap_vars2 = {
                'Dataset Emotions': 'emotion',
                'NRC Emotions': 'nrc',
                'LIWC Emotions': liwc_emotion_labels,
                'LIWC Tags': liwc_other_labels
            }

            var_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)

    with col_chart_var:

        #st.markdown('Detecting interference of variables can help to decide which data needs to handle.')

        if heatmap_option == 'Pearson':            

            if var_button_submited:                
                df_in = get_current_dataset(dset_selected_var)

                var1 = 'emotion'
                if heatmap_relation == 'Emotions X emotions':
                    var2 = 'emotion'
                elif heatmap_relation == 'Emotions X emotions (NRC)':
                    var2 = 'emotion_nrc'
                elif heatmap_relation == 'Emotions X emotions (LIWC)':
                    var2 = ['liwc_affect','liwc_posemo','liwc_negemo','liwc_anx','liwc_anger','liwc_sad']
                elif heatmap_relation == 'Emotions X tags (LIWC)':
                    var2 = ['liwc_negate','liwc_social','liwc_cogmech','liwc_percept','liwc_bio','liwc_relativ','liwc_relig','liwc_death']

                chart, df_heatmap = get_correlation_heatmap(dset_selected_var, df_in, var1, var2, '', "Emotions' Correlation", heatmap_min, heatmap_max)
                if chart:        
                    st.session_state.pearson_chart = chart
                    st.session_state.df_heat_map = df_heatmap
                    st.plotly_chart( (chart), use_container_width=True, theme=None )
            else:
                if 'pearson_chart' in st.session_state:
                    chart = st.session_state.pearson_chart
                    df_heatmap = st.session_state.df_heat_map
                    if chart:
                        st.plotly_chart( (chart), use_container_width=True, theme=None )

        else:
            if var_button_submited:
                df_jd_dc = get_current_dataset_jd_dc(dset_selected_var)

                var1_str_jd_dc = 'emotion'
                if heatmap_relation == 'Emotions X emotions (NRC)':
                    var2_str_jd_dc = 'nrc'
                elif heatmap_relation == 'Emotions X emotions (LIWC)':
                    var2_str_jd_dc = ['liwc_affect','liwc_posemo','liwc_negemo','liwc_anx','liwc_anger','liwc_sad']
                elif heatmap_relation == 'Emotions X tags (LIWC)':
                    var2_str_jd_dc = ['liwc_negate','liwc_social','liwc_cogmech','liwc_percept','liwc_bio','liwc_relativ','liwc_relig','liwc_death']


                chart_jd, chart_dc, df_jd = get_jaccard_dice_charts(df_jd_dc, var1_str_jd_dc, var2_str_jd_dc, z_min = heatmap_min, z_max = heatmap_max, chart_title = '') #, chart_width = 800, chart_height = 800)
                if heatmap_option == 'Jaccard':
                    chart = chart_jd                    
                elif heatmap_option == 'Dice':
                    chart = chart_dc
                    
                if chart:        
                    st.session_state.chart_jd = chart_jd
                    st.session_state.chart_dc = chart_dc
                    st.session_state.df_jd = df_jd
                    st.session_state.chart = chart
                    st.plotly_chart( (chart), use_container_width=False, theme=None )

            else:
                if 'chart_jd' in st.session_state:
                    chart_jd = st.session_state.chart_jd
                    chart_dc = st.session_state.chart_dc
                    chart = st.session_state.chart
                    df_jd = st.session_state.df_jd

                    st.plotly_chart( (chart), use_container_width=False, theme=None )


with tab_morphotree:
    last_dset_tree1 = ''
    last_emotion_tree1 = ''
    last_dset_tree2 = ''
    last_emotion_tree2 = ''

    #with st.form("morph_form"):

    col_filters_tree, col_chart_tree1, col_chart_tree2 = st.columns([1,3,3])

    with col_filters_tree:
        
        # Dataset 1
        dset_selected_tree1 = st.selectbox(
            label='Choose First Corpus', 
            key='DatasetTree1', 
            options=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
            label_visibility="visible",
            #on_change=rerun
        )
        st.session_state.dset_selected_tree1 = dset_selected_tree1

        # Filter dataset 1
        all_emotion_labels_tree1 = get_emotions_labels(dset_selected_tree1)

        #placeholder_emotions_tree1 = st.empty
        
        #if 'emotion_option_tree1' in st.session_state:
        #    emotion_option_tree1 = placeholder_emotions_tree1 = st.session_state.emotion_option_tree1
        #else:
            #emotion_option_tree1 = placeholder_emotions_tree1 = st.selectbox(
        emotion_option_tree1 = st.selectbox(
            label='Emotion',
            key='EmotionOptionTree1',
            options=all_emotion_labels_tree1,
            label_visibility="visible")
        st.session_state.emotion_option_tree1 = emotion_option_tree1
        
        # Dataset 2
        dset_selected_tree2 = st.selectbox(
            label='Choose Second Corpus', 
            key='DatasetTree2', 
            options=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
            label_visibility="visible"
        )

        # Filter dataset 2
        all_emotion_labels_tree2 = get_emotions_labels(dset_selected_tree2)

        emotion_option_tree2 = st.selectbox(
            label='Emotion',
            key='EmotionOptionTree2',
            options=all_emotion_labels_tree2,
            label_visibility="visible")
        
        #reload_choices4 = st.checkbox('Reload Last Choices', value=True, key='ReloadChoices4', label_visibility="visible")
        with st.form("morph_form"):
        
            morpho_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)
    

    with col_chart_tree1:

        #st.markdown('Morphological treemaps may express the vocabulary complexity.')

        if morpho_button_submited and (dset_selected_tree1 != last_dset_tree1 or emotion_option_tree1 != last_emotion_tree1):
            with st.spinner('Preparing the first treemap...'):
                chart1 = get_morphological_treemap(get_path_levels(), dset_selected_tree1, emotion_option_tree1, get_max_path_nodes())
                if chart1:
                    st.session_state.morpho_chart1 = chart1
                    st.plotly_chart( (chart1), use_container_width=True, theme=None )
                    last_dset_tree1 = dset_selected_tree1
                    last_emotion_tree1 = emotion_option_tree1
        else:
            if 'morpho_chart1' in st.session_state:
                chart1 = st.session_state.morpho_chart1
                st.plotly_chart( (chart1), use_container_width=True, theme=None )             

    with col_chart_tree2:

        #st.markdown('')

        if morpho_button_submited and (dset_selected_tree2 != last_dset_tree2 or emotion_option_tree2 != last_emotion_tree2):
            with st.spinner('Preparing the second treemap...'):
                chart2 = get_morphological_treemap(get_path_levels(), dset_selected_tree2, emotion_option_tree2, get_max_path_nodes())
                if chart2:
                    st.session_state.morpho_chart2 = chart2
                    st.plotly_chart( (chart2), use_container_width=True, theme=None )
                    last_dset_tree2 = dset_selected_tree2
                    last_emotion_tree2 = emotion_option_tree2
        else:
            if 'morpho_chart2' in st.session_state:
                chart2 = st.session_state.morpho_chart2
                st.plotly_chart( (chart2), use_container_width=True, theme=None )                        