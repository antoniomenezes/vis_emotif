import altair as alt
import pandas as pd
import streamlit as st
import time
import locale
import base64

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
#locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# É necessário instalar a biblioteca streamlit-aggrid, conforme abaixo
# pip install streamlit-aggrid
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import streamlit.components.v1 as components

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

#import plotly.plotly as py
from plotly.graph_objs import *

#from streamlit_plotly_events import plotly_events

pio.templates.default = "simple_white"

# Importing the dashboard auxiliary files
from dashboard_charts import *
from dashboard_data import *

from PIL import Image
logo_image = Image.open('./data/logo64.png')

alt.themes.enable("streamlit")

APP_TITLE = "Vis-EMotif"

st.set_page_config(
    page_title=APP_TITLE, page_icon=logo_image, layout='wide' # layout="centered" or "wide"
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

# Dependency treemap presents #paths higher than "dependency_tree_min_freq"
dependency_tree_min_freq = 10

@st.cache_data    
def get_emotions_labels(dataset_option):
    df_emotion_source_ = get_source_emotion()
    if dataset_option != 'All Datasets':
        df_emotion_source_ = df_emotion_source_[df_emotion_source_.source == dataset_option]

    # Filter dataset
    emo_labels = list(df_emotion_source_['emotion'].unique())
    return emo_labels


def show_grid(df, grid_height=600):
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_grid_options(domLayout='normal')

    st.write('Texts (limited to 100 lines)')

    gridOptions = gb.build()

    grid_response = AgGrid(
        df, 
        gridOptions=gridOptions,
        #height=grid_height, 
        width='100%',
        theme='streamlit',
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        enable_enterprise_modules=False
    )

# Main Dashboard Tabs
tab_home, tab_distfreq, tab_stats, tab_variables_rel, tab_morphotree = st.tabs(["Home", "Emotion Distribution", "Corpora Statistics", "Variables and Relationships", "Dependency Treemaps"])

# Home Tab
with tab_home:

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

    st.markdown("")
    st.markdown("This dashboard compares **five** benchmark datasets (corpora) used for textual emotion detection. Beyond statistics and linguistic features to be verified, there are other issues to examine: **imbalanced data**, **variable interference** and **data diversity/complexity**.")

    col_overview_table, col_overview_chart = st.columns([1,1])
    with col_overview_table:
        df_datasets_overview = get_datasets_overview()

        hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        st.table(df_datasets_overview)
        
    with col_overview_chart:
        with st.spinner("Preparing the datasets' overview chart..."):
            data_frame_count = get_df_all_source_counts()
            c_count = alt.Chart(data_frame_count, title="Datasets Sizes", width=600, height=250).mark_bar().encode(
                x=alt.X('counts:Q', title='#Lines'),
                y=alt.Y('source:N', title=None),
                color=alt.Color('source:N',title='Dataset',legend=None, scale=alt.Scale(scheme='category20')),
                tooltip = [
                    alt.Tooltip('source:N',title='Dataset'),
                    alt.Tooltip('counts:Q',format=",.0f")]
            )
            st.altair_chart(c_count, use_container_width=True)

    col_left, col_center, col_right = st.columns([1,10,1])
    with col_center:
        st.markdown('##### Guidelines');
        st.markdown('**"Emotion Distribution"** presents the balance of the data according to emotion classes. **"Corpora Statistics"** shows linguistical and descriptive metrics to explore the datasets. '+
                    'The **"Variables and Relationships"** tab helps to detect interference between variables. The **last one tab** presents dependency treemaps to examine the vocabulary complexity of the corpora.')
        st.caption("(*) relabeled: angry as anger, happy as joy, sad as sadness")


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
            max_textsize2 = st.slider(key = 'slider2', label = 'Maximum Text size', min_value = 0, max_value = max_length2, value = max_length2, step=10)

            emotionsfreq_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)

    with col_chart:
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

                st.session_state.distfreq_chart = c
                st.altair_chart(c, use_container_width=True)

        else:
            if 'distfreq_chart' in st.session_state:
                c = st.session_state.distfreq_chart
                st.altair_chart(c, use_container_width=True)
                

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
            options=['#Lines','#Duplicated Texts','Average #Sentences/Line','Average #Words/Line', 
                    'Entities','Average TTR/Line', 
                    'Avg. Flesch Reading Ease/Line'],
            help='Information about the feature will be below the chart',
            label_visibility="visible"
        )

        with st.form("stats_form"):
            options_slider = [x for x in range(0, max_length+1, 10)]
            options_slider.append(max_length)
            min_textsize, max_textsize = st.select_slider(key = 'slider1', label = 'Text size', value=(0, max_length), options=options_slider, label_visibility="visible")

            show_texts = st.checkbox('Show texts', value=False, key='ShowTexts', label_visibility="visible", disabled=False)

            stats_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)
        
        comments = get_comments(feature_selected)
        if len(comments)>0:
            st.markdown('**Tip**')
            for comment in comments:
                st.caption(comment)

    with col_charts_stats:
        if stats_button_submited:                

            with st.spinner("Preparing the chart..."):
                if feature_selected != 'Entities':
                    if 'tab_chart_plotly' in st.session_state:
                        del st.session_state['tab_chart_plotly']

                if feature_selected == '#Lines':
                    h = get_texts_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )

                elif feature_selected == '#Duplicated Texts':
                    if by_emotion:
                        h, df_h = get_duplicated_texts_chart(get_source_text_dup(dset_selected, min_textsize, max_textsize, by_emotion, 3), dset_selected, min_textsize, max_textsize, by_emotion)
                    else:
                        h, df_h = get_duplicated_texts_chart(get_source_text_dup(dset_selected, min_textsize, max_textsize, by_emotion, 30), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )

                elif feature_selected == 'Average #Sentences/Line':
                    h = get_sentences_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )

                elif feature_selected == 'Average #Words/Line':
                    h = get_words_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )

                elif feature_selected == 'Top Words':
                    h = None
                    if 'tab_chart' in st.session_state:
                        del st.session_state['tab_chart']

                elif feature_selected == '#Entities':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, 'n_ents', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )
                    
                elif feature_selected == 'Entities':
                    h = get_entities_chart(get_entities_data(dset_selected), dset_selected)
                    st.session_state.tab_chart_plotly = h
                    st.plotly_chart(h, use_container_width=True)
          
                elif feature_selected == 'Average TTR/Line':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, 'diversity_ttr', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )

                elif feature_selected == 'Average MTLD/Line':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, 'diversity_mtld', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)
                
                elif feature_selected == 'Avg. Flesch-Kincaid Grade Level/Line':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, 'flesch_kincaid_grade_level', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)
                
                elif feature_selected == 'Avg. Flesch Reading Ease/Line':
                    h = get_avg_feature_per_text_chart(get_source_emotion_stats_linesize(dset_selected, min_textsize, max_textsize), dset_selected, 'flesch_reading_ease', feature_selected, min_textsize, max_textsize, by_emotion)
                    st.altair_chart(h, use_container_width=False)

                    if show_texts:
                        show_grid( get_texts(dset_selected, feature_selected, min_textsize, max_textsize) )

                if 'tab_chart_plotly' not in st.session_state:
                    st.session_state.tab_chart = h
                
        else:
            if 'tab_chart' in st.session_state and 'tab_chart_plotly' not in st.session_state:
                h = st.session_state.tab_chart
                st.altair_chart(h, use_container_width=False)

            if 'tab_chart_plotly' in st.session_state:
                h = st.session_state.tab_chart_plotly
                st.plotly_chart(h, use_container_width=True)

            
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
                relations_available = ['Emotions + emotions','Emotions + emotions (NRC)','Emotions + emotions (LIWC)','Emotions + tags (LIWC)']
            else:
                relations_available = ['Emotions + emotions (NRC)','Emotions + emotions (LIWC)','Emotions + tags (LIWC)']    
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

        comments = get_comments(heatmap_option)
        if len(comments)>0:
            st.markdown('**Tip**')
            for comment in comments:
                st.caption(comment)


    with col_chart_var:

        if heatmap_option == 'Pearson':            

            if var_button_submited:           
                if 'chart_jd' in st.session_state:
                        del st.session_state['chart_jd']
                        del st.session_state['chart_jd_dc']
                        del st.session_state['df_jd']

                if 'chart_dc'  in st.session_state:                        
                        del st.session_state['chart_dc']

                df_in = get_dataset_for_pearson(dset_selected_var, heatmap_relation)

                chart, df_heatmap = get_correlation_heatmap(dset_selected_var, df_in, heatmap_relation, "Emotions' Correlation", heatmap_min, heatmap_max)
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

        # Heatmap Options (Jaccard/Dice)
        else:
            if var_button_submited:
                if 'pearson_chart' in st.session_state:
                        del st.session_state['pearson_chart']
                        del st.session_state['df_heat_map']

                df_jd_dc = get_dataset_jd_dc(dset_selected_var, heatmap_relation)

                var1_str_jd_dc = 'emotion'
                var2_str_jd_dc = ''

                if heatmap_relation == 'Emotions X emotions (NRC)':
                    var2_str_jd_dc = 'nrc'
                elif heatmap_relation == 'Emotions X emotions (LIWC)':
                    var2_str_jd_dc = liwc_emotion_labels 
                elif heatmap_relation == 'Emotions X tags (LIWC)':
                    var2_str_jd_dc = liwc_other_labels 

                chart_jd, chart_dc, df_jd = get_jaccard_dice_charts(df_jd_dc, var1_str_jd_dc, var2_str_jd_dc, z_min = heatmap_min, z_max = heatmap_max, chart_title = '') #, chart_width = 800, chart_height = 800)
                if heatmap_option == 'Jaccard':
                    chart = chart_jd                    
                elif heatmap_option == 'Dice':
                    chart = chart_dc
                    
                if chart:        
                    st.session_state.chart_jd = chart_jd
                    st.session_state.chart_dc = chart_dc
                    st.session_state.df_jd = df_jd
                    st.session_state.chart_jd_dc = chart
                    st.plotly_chart( (chart), use_container_width=False, theme=None )                    

            else:
                if 'chart_jd' in st.session_state:
                    chart_jd = st.session_state.chart_jd
                    chart_dc = st.session_state.chart_dc
                    chart = st.session_state.chart_jd_dc
                    df_jd = st.session_state.df_jd
                    st.plotly_chart( (chart), use_container_width=False, theme=None )


with tab_morphotree:
    last_dset_tree1 = ''
    last_emotion_tree1 = ''
    last_dset_tree2 = ''
    last_emotion_tree2 = ''

    col_filters_tree, col_chart_tree1, col_chart_tree2 = st.columns([1,3,3])

    with col_filters_tree:
        
        # Dataset 1
        dset_selected_tree1 = st.selectbox(
            label='Choose First Corpus', 
            key='DatasetTree1', 
            options=['CARER','GoEmotions','ISEAR','SemEval2019 Task3','TEC'],
            label_visibility="visible",
        )
        st.session_state.dset_selected_tree1 = dset_selected_tree1

        # Filter dataset 1
        all_emotion_labels_tree1 = get_emotions_labels(dset_selected_tree1)

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
        
        with st.form("morph_form"):        
            morpho_button_submited = st.form_submit_button(label="Plot", help=None, on_click=None, use_container_width=True)

        st.markdown('**Tip**')
        st.caption("POS Tags paths with frequency greater than "+str(dependency_tree_min_freq)+" will be plotted.")
    

    with col_chart_tree1:

        if morpho_button_submited and (dset_selected_tree1 != last_dset_tree1 or emotion_option_tree1 != last_emotion_tree1):
            with st.spinner('Preparing the first treemap...'):
                chart1 = get_dependency_treemap(dset_selected_tree1, emotion_option_tree1, dependency_tree_min_freq)
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

        if morpho_button_submited and (dset_selected_tree2 != last_dset_tree2 or emotion_option_tree2 != last_emotion_tree2):
            with st.spinner('Preparing the second treemap...'):
                chart2 = get_dependency_treemap(dset_selected_tree2, emotion_option_tree2, dependency_tree_min_freq)
                if chart2:
                    st.session_state.morpho_chart2 = chart2

                    st.plotly_chart( (chart2), use_container_width=True, theme=None )
                    last_dset_tree2 = dset_selected_tree2
                    last_emotion_tree2 = emotion_option_tree2

        else:
            if 'morpho_chart2' in st.session_state:
                chart2 = st.session_state.morpho_chart2

                st.plotly_chart( (chart2), use_container_width=True, theme=None )  
