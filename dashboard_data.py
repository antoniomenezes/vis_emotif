import pandas as pd
import streamlit as st

DATA_DIR = './data'
dataset_dir = DATA_DIR

GO_EMOTIONS_DIR = dataset_dir
HF_DIR = dataset_dir
ISEAR_DIR = dataset_dir
SAIF_MOHAMMAD_DIR = dataset_dir
SEMEVAL19_DIR = dataset_dir

dict_emotions_colors = {
    "anger":"#ff0000",
    "fear":"#009600",
    "joy":"#ffff54", # before: "#ffe778",
    "love":"#ff00ff",
    "sadness":"#5151ff",
    "surprise":"#59bdff",
    "admiration":"#00b400",
    "amusement":"#af00ff",
    "annoyance":"#ff8c8c",
    "approval":"#77f400",
    "caring":"#ffb3d4",
    "confusion":"#620f9a",
    "curiosity":"#11b3d4",
    "desire":"#f51e7b",
    "disappointment":"#9a4781",
    "disapproval":"#a4b8ff",
    "disgust":"#ff54ff",
    "embarrassment":"#620f3d",
    "excitement":"#ffe854", #ecstasy
    "gratitude":"#f5ddb1",
    "grief":"#2e2e3f", # before: "#0000c8",
    "nervousness":"#6a57a0",
    "neutral":"#cccccc",
    "optimism":"#fee1aa",
    "pride":"#d4551b",
    "realization":"#00ffa6",
    "relief":"#8abdd6",
    "remorse":"#c5aeff",
    "guilt":"#8cc68c", #apprehension
    "shame":"#392e50", 
    "others":"#7f7f7f"
}

ge_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
nrc_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']


@st.cache_data
def get_datasets_overview():
    dict_overview = {
        'Dataset' : ['CARER', 'GoEmotions', 'ISEAR', 'SemEval 2019 Task 3', 'TEC'],
        'Year' : [2018, 2020, 1997, 2019, 2012],
        '#Emotions' : [6, 28, 7, 4, 6],
        'Domain' : ['Tweets', 'Reddit comments', 'Description of situations', 'Dialogs', 'Tweets'],
        'Annotation Task' : ['Automatic with hashtags','Manual','Questionnaire','Collected from a pool','Automatic with hashtags']
    }
  
    # creating a Dataframe object 
    df_overview = pd.DataFrame(dict_overview)
    return df_overview

#@st.cache_data
def get_color(emotion):
    return dict_emotions_colors[emotion]

#@st.cache_data
def get_emotions_colors(emotions):
    emotions_colors =[get_color(emotion) for emotion in emotions]
    #emotions_colors = [item[1] for item in dict_emotions_colors.items() if item[0] in emotions]
    return emotions_colors

@st.cache_data
def get_source_emotion():
    if 'df_source_emotion' in st.session_state:
        df_source_emotion = st.session_state.df_source_emotion
    else:
        df_source_emotion = pd.read_csv(dataset_dir+'/data_source_emotion.csv', encoding='UTF-8', sep='|')
        df_source_emotion = df_source_emotion[['source','emotion','counts']]
        st.session_state.df_source_emotion = df_source_emotion
    return df_source_emotion

@st.cache_data
def get_source_emotion_length():
    df_emotion_source_length = pd.read_csv(dataset_dir+'/emotion_source_length.csv', encoding='UTF-8', sep='|')
    return df_emotion_source_length

@st.cache_data
def get_source_emotion_stats():
    df_text_emotion_source_stats = pd.read_csv(dataset_dir+'/emotion_source_stats.csv', encoding='UTF-8', sep='|')
    return df_text_emotion_source_stats

@st.cache_data
def get_source_text_dup():
    df_text_source_dup = pd.read_csv(dataset_dir+'/data_text_source_dup.csv', encoding='UTF-8', sep='|')
    return df_text_source_dup

@st.cache_data
def get_path_levels():
    df_path_levels = pd.read_csv(dataset_dir+'/source_emotion_path_levels.csv', encoding='UTF-8', sep='|')
    return df_path_levels

@st.cache_data
def get_max_path_nodes():
    df_path_levels = get_path_levels()
    return df_path_levels.path_length.max()

@st.cache_data
def get_df_stats(datasetname):
    if datasetname == 'CARER':
        df = pd.read_csv(HF_DIR+'/carer_no_text_stats.csv', encoding='UTF-8', sep='|')
    elif datasetname == 'GoEmotions':
        df = pd.read_csv(GO_EMOTIONS_DIR+'/goemotions_no_text_stats.csv', encoding='UTF-8', sep='|')
    elif datasetname == 'ISEAR':
        df = pd.read_csv(ISEAR_DIR+'/isear_no_text_stats.csv', encoding='UTF-8', sep='|')
    elif datasetname == 'TEC':
        df = pd.read_csv(SAIF_MOHAMMAD_DIR+'/tec_no_text_stats.csv', encoding='UTF-8', sep='|')
    elif datasetname == 'SemEval2019 Task3':
        df = pd.read_csv(SEMEVAL19_DIR+'/se_no_text_stats.csv', encoding='UTF-8', sep='|')
    else:
        df = pd.DataFrame()
    return df

@st.cache_data
def get_df_all_source_counts():
    df_source = pd.read_csv(dataset_dir+'/data_source.csv', encoding='UTF-8', sep='|')
    df_source = df_source[['source','counts']]
    return df_source


@st.cache_data
def get_current_dataset_jd_dc(dataset_option):
    if dataset_option == 'CARER':
        df = pd.read_csv(dataset_dir+'/emotion_huggingface_jd.csv', encoding='UTF-8', sep='|')
    elif dataset_option == 'GoEmotions':
        df = pd.read_csv(dataset_dir+'/goemotions_jd.csv', encoding='UTF-8', sep='|')
    elif dataset_option == 'ISEAR':
        df = pd.read_csv(dataset_dir+'/isear_jd.csv', encoding='UTF-8', sep='|')
    elif dataset_option == 'TEC':
        df = pd.read_csv(dataset_dir+'/saif_jd.csv', encoding='UTF-8', sep='|')
    elif dataset_option == 'SemEval2019 Task3':
        df = pd.read_csv(dataset_dir+'/se_jd.csv', encoding='UTF-8', sep='|')
    else:
        df = pd.DataFrame()
    return df
